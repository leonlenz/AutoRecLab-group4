import pickle
import random
import shutil
from pathlib import Path

from anytree import PreOrderIter

from config import CONFIG_PATH, Config

from treesearch.interpreter import Interpreter
from treesearch.minimal_agent import MinimalAgent
from treesearch.node import Node
from utils.log import _ROOT_LOGGER
from utils.path import mkdir
from viz import render_trees

logger = _ROOT_LOGGER.getChild("treesearch")


class TreeSearch:
    def __init__(self, user_request: str, config: Config) -> None:
        self._user_request = user_request
        self._config = config
        self._draft_nodes: list[Node] = []
        self._out_dir = mkdir(Path(config.out_dir))
        workspace_pth = mkdir(self._out_dir / "workspace").resolve()
        self._workspace = str(workspace_pth)
        self._checkpoint_dir = mkdir(self._out_dir / "checkpoint")

        shutil.copy(CONFIG_PATH, self._out_dir)

        self._minimal_agent = MinimalAgent(
            self._task_desc,
            self._config,
            evaluation_metrics=self._config.agent.evaluation_metrics,
        )
        self._interpreter = Interpreter(self._workspace, self._config.exec.timeout)

    async def _async_init(self):
        await self._minimal_agent._async_init()

    @property
    def all_nodes(self):
        return [n for root in self._draft_nodes for n in PreOrderIter(root)]

    @property
    def good_nodes(self):
        return list(filter(lambda n: not n.is_buggy, self.all_nodes))

    @property
    def buggy_nodes(self):
        return list(filter(lambda n: n.is_buggy, self.all_nodes))

    @property
    def best_good_node(self):
        good_nodes = self.good_nodes
        good_nodes.sort(key=lambda n: n.score.score, reverse=True)
        return good_nodes[0]

    @property
    def best_buggy_node(self):
        buggy_nodes = self.buggy_nodes
        buggy_nodes.sort(key=lambda n: n.score.score, reverse=True)
        return buggy_nodes[0]

    def select_next_node(self) -> Node:
        if (
            len(self.buggy_nodes) > 0
            and random.random() < self._config.treesearch.debug_prob
            or len(self.good_nodes) == 0
        ):
            if random.random() < self._config.treesearch.epsilon:
                logger.info("Selecting random buggy node for debugging...")
                nodes = self.buggy_nodes
                weights = [1 / (len(n.children) + 1) for n in nodes]
                return random.choices(nodes, weights=weights, k=1)[0]
            else:
                logger.info("Selecting best buggy node for debugging...")
                return max(self.buggy_nodes, key=lambda n: n.score.score * (1 / (len(n.children) + 1)))

        if random.random() < self._config.treesearch.epsilon:
            nodes = self.good_nodes
            weights = [1 / (len(n.children) + 1) for n in nodes]
            return random.choices(nodes, weights=weights, k=1)[0]
        else:
            return max(self.good_nodes, key=lambda n: n.score.score * (1 / (len(n.children) + 1)))

    async def run(self):
        logger.info("Starting tree search...")
        # Step 1: Generate draft nodes:
        for i in range(self._config.treesearch.num_draft_nodes):
            logger.info(
                f"Generating draft node {i + 1}/{self._config.treesearch.num_draft_nodes}"
            )
            draft_node = await self._minimal_agent._draft()
            await self.exec_node(draft_node)
            self._draft_nodes.append(draft_node)

        for i in range(self._config.treesearch.max_iterations):
            logger.info(
                f"Treesearch iteration {i + 1}/{self._config.treesearch.max_iterations}"
            )
            parent_node = self.select_next_node()

            if parent_node.is_buggy:
                child_node = await self._minimal_agent._debug(parent_node)
            else:
                child_node = await self._minimal_agent._improve(parent_node)

            await self.exec_node(child_node)

            if child_node.score.is_satisfactory:
                logger.info("Found satisfactory node:")
                self.save()
                await self.finalize_search(child_node)
                return

        self.save()

        logger.warning("Found no satisfactory node; Using best node instead...")

        if len(self.good_nodes) == 0:
            logger.warning("No good nodes found; Using best buggy node...")
            best_node = self.best_buggy_node
        else:
            best_node = self.best_good_node
        await self.finalize_search(result_node=best_node)

    async def exec_node(self, node: Node) -> Node:
        exec_result = self._interpreter.run(node.code)
        logger.debug(exec_result)

        node_dir = mkdir(self._checkpoint_dir / node.id)
        (node_dir / "code.py").write_text(node.code)
        (node_dir / "out.log").write_text("".join(exec_result.term_out))
        (node_dir / "exec_result.pkl").write_bytes(pickle.dumps(exec_result))

        # Move all generated files from the workspace to checkpoint for this node
        workspace_dir = Path(self._workspace)
        working_dir = workspace_dir / "working"
        
        # Collect files from workspace (excluding runfile.py and working dir)
        generated_files = [
            item for item in workspace_dir.iterdir()
            if item.name not in ("runfile.py", "working") and not item.name.startswith(".")
        ]
        
        # Also collect files from working subdirectory if it exists
        if working_dir.exists():
            generated_files.extend(list(working_dir.iterdir()))
        
        if generated_files:
            generated_dir = mkdir(node_dir / "generated")
            for item in generated_files:
                try:
                    shutil.move(str(item), str(generated_dir / item.name))
                    logger.info(f"Moved {item.name} to checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to move {item.name}: {e}")

        await self._minimal_agent.score_code(node, exec_result)
        return node

    async def finalize_search(self, result_node: Node):
        self._interpreter.cleanup_session()
        logger.info("Final response:")
        print(await self._minimal_agent._summarize(self._user_request, result_node))

    @property
    def _task_desc(self) -> str:
        task_desc = """ You are an expert recommender systems research assistant who is looking to help the user with their requests.
                    The user has some idea and you want to conduct creative experiments to gain scientific insights.
                    Your aim is to run experiments to gather sufficient results to report back to the user.
                    The idea is:\n
                    """
        task_desc += self._user_request
        return task_desc

    def save(self):
        logger.info("Generating tree visualization...")
        tree_render_dir = mkdir(self._out_dir / "tree_render")
        render_trees(self._draft_nodes, tree_render_dir)

        with open(self._out_dir / "save.pkl", "wb") as f:
            logger.info(f"SAVING {len(self._draft_nodes)}.....")
            pickle.dump(self._draft_nodes, f)
