import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Self

from anytree import NodeMixin

from treesearch.interpreter import ExecutionResult
from treesearch.metric import MetricValue
from treesearch.type_checker import TypeCheckResult
from treesearch.utils.response import trim_long_string


@dataclass
class Requirement:
    description: str
    is_fulfilled = False
    feedback: Optional[str] = None


@dataclass
class NodeScore:
    score: float = 0.0
    feedback: str = ""
    is_satisfactory: bool = False


@dataclass(eq=False, kw_only=True)
class Node(NodeMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    plan: str = field(default="")  # type: ignore
    overall_plan: str = field(default="")  # type: ignore
    code: str = field(default="")  # type: ignore
    plot_code: str = field(default=None)  # type: ignore
    plot_plan: str = field(default=None)  # type: ignore

    # ---- general attr ----
    _parent: Optional[Self] = field(default=None)
    step: int = field(default=None)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ctime: float = field(default_factory=lambda: time.time())
    exp_results_dir: str = field(default=None)  # type: ignore
    score: NodeScore = field(default_factory=NodeScore)

    # ---- execution info ----
    _term_out: list[str] = field(default=None)  # type: ignore
    exec_time: float = field(default=None)  # type: ignore

    # ---- parsing info ----
    parse_metrics_plan: str = field(default="")
    parse_metrics_code: str = field(default="")
    # parse_exec_result: ExecutionResult = field(default=None, kw_only=True)
    parse_term_out: Optional[list[str]] = field(default=None)
    parse_exc_type: str | None = field(default=None)
    parse_exc_info: dict | None = field(default=None)
    parse_exc_stack: list[tuple] | None = field(default=None)

    # ---- plot execution info ----
    plot_term_out: list[str] = field(default=None)  # type: ignore
    plot_exec_time: float = field(default=None)  # type: ignore

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None)  # type: ignore
    metric: MetricValue = field(default=None)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None)  # type: ignore
    is_buggy_plots: Optional[bool] = field(default=None)
    requirements: list[Requirement] = field(default_factory=list)

    # ---- plotting ----
    plot_data: dict = field(default_factory=dict)
    plots_generated: bool = field(default=False)
    plots: list[str] = field(
        default_factory=list, kw_only=False
    )  # Relative paths for visualization
    plot_paths: list[str] = field(
        default_factory=list, kw_only=False
    )  # Absolute paths for programmatic access

    # ---- VLM feedback ----
    plot_analyses: list[str] = field(default_factory=list, kw_only=False)
    vlm_feedback_summary: list[str] = field(default_factory=list, kw_only=False)
    datasets_successfully_tested: list[str] = field(default_factory=list, kw_only=False)

    # ---- execution time feedback ----
    exec_time_feedback: str = field(default="")

    # ---- type checking info ----
    type_check_attempts: int = field(default=0)
    type_check_passed: bool = field(default=False)
    type_check_results: list[TypeCheckResult] = field(default_factory=list)

    # ---- ablation study ----
    ablation_name: Optional[str] = field(default=None)

    # ---- hyperparam tuning ----
    hyperparam_name: Optional[str] = field(default=None)

    # ---- seed node ----
    is_seed_node: bool = field(default=False)
    is_seed_agg_node: bool = field(default=False)

    @property
    def name(self) -> str:
        short_id = f"{self.id[:4]}...{self.id[-4:]}"
        if len(self.plan) > 0:
            plan_max_chars = 25
            dots = "..." if len(self.plan) > plan_max_chars else ""
            return f"{__class__.__name__}({short_id},\n{self.plan[:plan_max_chars]}{dots}\nbuggy={self.is_buggy}\nscore={self.score.score})"
        else:
            return f"{__class__.__name__}({short_id}\nbuggy={self.is_buggy}\nscore={self.score.score})"

    def __post_init__(self):
        self.parent = self._parent

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        # Create a new instance with copied attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except parent and children to avoid circular references
        for k, v in self.__dict__.items():
            if k not in ("parent", "children"):
                setattr(result, k, copy.deepcopy(v, memo))

        # Handle parent and children separately
        result.parent = self.parent  # Keep the same parent reference
        result.children = set()  # Start with empty children set

        return result

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        # Ensure id is included in the state
        if hasattr(self, "id"):
            state["id"] = self.id
        return state

    def __setstate__(self, state):
        """Set state during unpickling"""
        # Ensure all required attributes are present
        self.__dict__.update(state)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time

    def absorb_plot_exec_result(self, plot_exec_result: ExecutionResult):
        """Absorb the result of executing the plotting code from this node."""
        self.plot_term_out = plot_exec_result.term_out
        self.plot_exec_time = plot_exec_result.exec_time

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore
