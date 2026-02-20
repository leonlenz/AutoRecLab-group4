import json
import random
from pathlib import Path
from typing import Any, Optional

import humanize

from config import Config
from treesearch.function_specs import (
    CodeRequirements,
    PlanAndCode,
    ReviewFunction,
    ScoreCode,
    SelectDatasets,
)
from treesearch.interpreter import ExecutionResult
from treesearch.llm.query import MCPConnection, Prompt, Query
from treesearch.node import Node, NodeScore, Requirement
from treesearch.utils.available_datasets import get_datasets_table
from treesearch.mcp.docs_search_server import VECTOR_STORE_NAMES
from treesearch.utils.response import strip_markdown_fences
from utils.log import _ROOT_LOGGER
from utils.path import mkdir

logger = _ROOT_LOGGER.getChild("nodeAgent")


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        memory_summary=None,
        evaluation_metrics=None,
        stage_name=None,
    ):
        logger.info("Initializing agent...")
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        self._out_dir = mkdir(Path(cfg.out_dir))
        logger.info("Agent initialized!")

        # Setup MCP connections for documentation search
        self._mcp_docs = MCPConnection(
            name="docs_search",
            connection={
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "treesearch.mcp.docs_search_server"],
            },
        )

    async def _async_init(self):
        self.selected_datasets = await self._select_datasets()
        await self._set_code_requirements()
        (self._out_dir / "code_requirements.json").write_text(
            json.dumps(self.code_requirements)
        )

    @property
    def _prompt_environment(self):
        pkgs = [
            "Primary: omnirec==0.2.0",
            "numpy==1.26.4",
            "numba==0.58.1",
            "pandas==2.3.2",
            "scipy==1.16.2",
            "scikit-learn==1.7.1",
            "lenskit==2025.6.2",
            "matplotlib==3.10.7",
        ]
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use the following machine learning packages: {pkg_str}. You MUST use these libraries as much as possible instead of implementing from scratch."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "Implementation Guidelines:",
            f"1. Framework: Use OmniRec exclusively (wraps Lenskit, RecBole, RecPack, Elliot, etc.). Search these docs when unsure: {VECTOR_STORE_NAMES}",
            f"2. Datasets: Use only: {', '.join(self.selected_datasets)}",
            "3. Code Structure:",
            "   - Single-file Python script with `if __name__ == '__main__':`",
            "   - Keep simple - use only well-documented APIs",
            "4. Environment Setup:",
            "   - Create working directory: `working_dir = os.path.join(os.getcwd(), 'working'); os.makedirs(working_dir, exist_ok=True)`",
            f"   - Complete execution within {humanize.naturaldelta(self.cfg.exec.timeout)}",
            "5. Data Saving:",
            "   - Store all metrics/losses in `experiment_data` dict",
            "   - Save: `np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)`",
            "6. Evaluation:",
            f"   - Metrics: {', '.join(self.evaluation_metrics) if self.evaluation_metrics else 'Choose appropriate metrics'}",
            "   - Print metrics during execution for monitoring",
            "7. API Verification (CRITICAL):",
            "   - Check constructor signatures before use",
            "   - Verify object attributes exist (e.g., SplitData structure)",
            "   - Use only public APIs (no underscore-prefixed methods)",
        ]

        if self.cfg.agent.k_fold_validation > 1:
            impl_guideline.append(
                f"9. Validation: Use {self.cfg.agent.k_fold_validation}-fold cross-validation if appropriate."
            )

        return {"Implementation guideline": impl_guideline}

    async def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a meticulous Recommender Systems Engineer and Researcher. "
                "Your task is to: 1) Research the correct API usage for the given task, "
                "2) Design a baseline, and 3) Implement it. "
                "Do not implement code based on memory; always verify method signatures via the provided search tool."
            ),
            "Research task": self.task_desc,
            "Code Requirements": self.code_requirements
            if hasattr(self, "code_requirements")
            else "",
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design. ",
                "The solution sketch should be 6-10 sentences. ",
                "Don't suggest to do EDA.",
                "Make sure to use the provided dataset(s).",
                "",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        print("[cyan]--------------------------------[/cyan]")
        print("[cyan]self.task_desc[/cyan]")
        print("[cyan]" + self.task_desc + "[/cyan]")
        print("[cyan]--------------------------------[/cyan]")

        print("MinimalAgent: Getting plan and code")
        plan, code = await self.plan_and_code_query(prompt)
        print("MinimalAgent: Draft complete")
        return self._new_node(plan, code)

    async def _debug(self, parent_node: Node) -> Node:
        # Format node scores for the prompt
        score_info = ""
        if hasattr(parent_node, "score") and parent_node.score:
            score_info = f"""
                Previous Implementation Scores:
                - Score: {parent_node.score.score * 100:.1f}%
                - Is Satisfactory: {parent_node.score.is_satisfactory}
                - Feedback: {parent_node.score.feedback}
                """

        # Enhanced bug analysis for more helpful feedback
        bug_analysis = (
            parent_node.analysis
            if parent_node.analysis
            else "Bug analysis not available"
        )
        if parent_node.is_buggy and parent_node.analysis:
            enhanced_bug_info = f"""
                Bug Analysis:
                {bug_analysis}

                This indicates the code failed to execute properly. Focus on addressing the specific error mentioned above.
                """
        else:
            enhanced_bug_info = f"Previous implementation had issues: {bug_analysis}"

        prompt: Any = {
            "Introduction": (
                "You are a Senior Debugging Engineer. Your goal is to resolve execution errors in a recommender system script. You must treat the 'Execution output' as truth and the 'Previous implementation' as potentially fundamentally flawed API-wise. Do not assume the previous code's use of libraries was correct."
            ),
            "Research task": self.task_desc,
            "Previous (buggy) implementation": parent_node.code,
            "Execution output": parent_node.term_out,
            "Bug Analysis & Scoring": enhanced_bug_info + score_info,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "1. ERROR DIAGNOSIS: Analyze the 'Execution output' specifically for API errors (AttributeError, TypeError, etc.).",
                f"2. DOCUMENTATION VERIFICATION: If the error involves {VECTOR_STORE_NAMES} you MUST search the documentation for the correct class/method signature before writing the fix.",
                "3. EXPLAIN THE FIX: Write 3-5 sentences describing the root cause and the verified solution. Cite the documentation if an API change was made.",
                "4. DO NOT GUESS: If the documentation does not show the method you need, search for alternatives or 'examples' in the MCP server.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = await self.plan_and_code_query(prompt)
        return self._new_node(plan, code, parent_node)

    async def _improve(self, parent_node: Node) -> Node:
        # Format node scores for the prompt
        score_info = ""
        if hasattr(parent_node, "score") and parent_node.score:
            score_info = f"""
                Previous Implementation Scores:
                - Score: {parent_node.score.score * 100:.1f}%
                - Is Satisfactory: {parent_node.score.is_satisfactory}
                - Feedback: {parent_node.score.feedback}
                """

        prompt: Any = {
            "Introduction": (
                "You are an experienced recommender systems researcher. You are provided with a previously developed "
                "implementation. Your task is to improve it to meet the research task requirements and address any issues identified in the 'Performance Analysis'."
            ),
            "Research task": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Performance Analysis & Scoring": score_info,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": parent_node.code,
        }

        prompt["Instructions"] |= {
            "Refactoring & Compliance Guidelines": [
                "1. ANALYZE FEEDBACK: Map each piece of feedback from the 'Performance Analysis' to a specific line or block in your previous code.",
                "2. CONSULT THE SOURCE: For every requirement that was marked as 'unsatisfactory,' search the documentation for the 'canonical' way to implement that feature.",
                "3. REFACTOR, DON'T PATCH: Do not just add 'if' statements to hide errors. Rewrite the implementation to align with the framework's intended API usage as found in the docs.",
                "4. PRESERVE LOGIC: Ensure the research task's scientific logic remains intact while updating the code structure to meet the requirements.",
            ]
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = await self.plan_and_code_query(prompt)
        return self._new_node(plan, code, parent_node)

    def _new_node(self, plan: str, code: str, parent: Optional[Node] = None):
        return Node(
            plan=plan,
            code=code,
            _parent=parent,
            requirements=[Requirement(r) for r in self.code_requirements],
        )

    async def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        plan_and_code_result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                f"You are a Senior Recommender Systems Engineer specializing in the OmniRec library. "
                f"Available documentation (OmniRec and libraries that OmniRec can use): {VECTOR_STORE_NAMES}.\n"
                "\n"
                "Search documentation to verify API details. Process:\n"
                "1. Identify needed components → 2. Search + verify each → 3. Document findings → 4. Implement\n"
                "\n"
                "Verify in documentation:\n"
                "- Function signatures (parameter names, types, valid ranges)\n"
                "- Object attributes (use public APIs only, not _private)\n"
                "- Data structures and return types\n"
                "\n"
                "In 'nl_text', include '## Documentation Verified' section listing all verified methods.\n"
                "Search for examples and Verify critical details in documentation."
            )
            .run(prompt, PlanAndCode)
        )

        nl_text = plan_and_code_result.nl_text
        code = strip_markdown_fences(plan_and_code_result.code)
        return nl_text, code

    async def _select_datasets(self) -> list[str]:
        """Select appropriate datasets for the research task using LLM."""
        prompt: Prompt = {
            "Instruction:": (
                f"You are a recommender system researcher selecting datasets for a research task.\n\n"
                f"Research task:\n{self.task_desc}\n\n"
                "Instructions:\n"
                "1. Check if the research task specifies any datasets\n"
                "2. If specified, select those datasets; otherwise choose appropriate ones from the list below\n"
                "3. Return only a list of dataset identifiers\n\n"
                f"Available datasets:\n{get_datasets_table()}"
            )
        }
        result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "Search OmniRec documentation for dataset characteristics and usage patterns if needed."
            )
            .run(prompt, SelectDatasets)
        )
        return result.selected_datasets

    async def _set_code_requirements(self):
        logger.info("Engineering code requirements...")
        requirements_prompt = f"""
        You are an expert recommender systems researcher defining experiment requirements.

        Research task: {self.task_desc}
        Selected datasets: {self.selected_datasets}

        Generate requirements that specify critical aspects of the experiment that must be fulfilled.

        PRINCIPLES:

        1. Minimal and necessary: Only include requirements essential for THIS specific research task
        - If removing a requirement would make the experiment fail or meaningless, keep it
        - If a requirement is general best practice but not necessary for this task, exclude it

        2. Abstraction: State objectives and constraints at an appropriate level
        - Avoid excessive implementation details (exact formulas, nested conditional logic, code-level instructions)
        - Include critical technical specifications where they matter (framework to use, specific datasets, evaluation metrics, split ratios)

        3. Atomicity: Each requirement should test one distinct aspect of the experiment

        4. Coverage: Include requirements for all essential aspects:
        - Data loading and preprocessing
        - Experimental methodology (data splitting, reproducibility requirements)
        - Model/algorithm selection and configuration
        - Training procedures
        - Evaluation methodology and metrics
        - Critical outputs and results

        Include both technical requirements (correct tool/API usage) and conceptual requirements (methodologically sound experiment design), but keep it as minimal as possible.
        """
        requirements_result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "Reference documentation for OmniRec framework and dataset details if needed to ensure requirements are feasible. Prioritize implementation guides and API references."
            )
            .run(requirements_prompt, CodeRequirements)
        )
        if len(requirements_result.requirements) == 0:
            self.code_requirements = "No specific requirements provided."
        else:
            self.code_requirements = requirements_result.requirements

        # Requirements reflection round
        reflection_prompt = f"""
        Quality review: Ensure requirements are minimal, atomic, and sufficient for the research task.

        Research task: {self.task_desc}
        Generated requirements: {self.code_requirements}

        REVIEW CRITERIA:

        1. Necessity: Is each requirement essential for THIS research task?
        - Keep only requirements whose absence would make the experiment fail or invalid
        - Remove general best practices, optimizations, or requirements not relevant to this specific task

        2. Appropriate detail level: 
        - Remove excessive implementation details (step-by-step procedures, exact formulas, nested logic)
        - Retain critical technical specifications (which framework, which metrics, which methodology)

        3. Atomicity: Each requirement tests one distinct aspect - split compound requirements

        4. Coverage: All critical aspects covered for this specific task:
        - Technical correctness (proper use of tools/APIs, data handling, reproducibility)
        - Conceptual correctness (valid experimental design for the research question)

        Refine the list: remove unnecessary requirements, simplify over-detailed ones, split compound ones, add missing critical aspects.
        """
        reflection_result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "Verify requirements against documented best practices. Reference documentation to confirm technical details are correct."
            )
            .run(reflection_prompt, CodeRequirements)
        )
        if len(reflection_result.requirements) == 0:
            self.code_requirements = "No specific requirements provided."
        else:
            self.code_requirements = reflection_result.requirements
        logger.info("Done.")

    async def score_code(self, node: Node, exec_result: ExecutionResult) -> Node:
        """Analyze execution results using both review function spec and scoring system."""
        node.absorb_exec_result(exec_result)

        # First, use the review_func_spec for buggy node identification
        review_prompt = {
            "Introduction": (
                "You are an expert recommender systems researcher conducting a code review. "
                "Your task is to evaluate whether the code execution was successful or contains bugs. "
                "Focus on identifying execution failures, errors, or other issues that would prevent the code from working properly."
            ),
            "Research Task": self.task_desc,
            "Implementation": node.code,
            "Execution Output": node.term_out
            if node.term_out
            else "No output generated",
            "Instructions": [
                "Carefully analyze the execution output for signs of bugs or failures:",
                "- Syntax errors, import errors, or runtime exceptions",
                "- Missing required outputs or metrics",
                "- Execution timeouts or crashes",
                "- Incorrect or nonsensical results",
                "If there's a bug, provide a clear summary of the issue and suggest how to fix it.",
                "If the execution was successful, leave the summary empty.",
            ],
        }

        bug_feedback = ""

        try:
            review_result = (
                await Query()
                .with_mcp(self._mcp_docs)
                .with_system(
                    "Search for usage examples in documentation when diagnosing API-related bugs. Look for common error patterns and correct API usage."
                )
                .run(review_prompt, ReviewFunction)
            )

            # Update node with review results
            node.is_buggy = review_result.is_bug
            node.analysis = review_result.summary

            if node.is_buggy:
                logger.info(f"Node identified as buggy: {node.analysis}")
                # Create more helpful feedback for buggy nodes
                bug_feedback = f"""EXECUTION FAILURE DETECTED:

                    {node.analysis}

                    NEXT STEPS FOR DEBUGGING:
                    - Review the error message above carefully
                    - Check for missing imports or incorrect package names
                    - Verify variable names and function calls
                    - Ensure all required data files are accessible
                    - Consider simplifying the code to isolate the issue

                    This implementation failed execution. Focus on resolving the error before optimizing."""

        except Exception as e:
            logger.error(f"Error in code review: {e}")
            # Fallback: mark as buggy if analysis fails
            node.is_buggy = True
            node.analysis = f"Review analysis failed: {str(e)}"

            bug_feedback = f"""ANALYSIS SYSTEM ERROR:

                The automated review system encountered an error: {str(e)}

                MANUAL REVIEW REQUIRED:
                - Check the execution output manually for obvious errors
                - Look for common issues like import errors, syntax errors, or missing dependencies
                - Verify that all required packages are installed
                - Test the code in smaller chunks to isolate any problems

                This implementation scored 0% due to analysis failure. Manual debugging recommended."""

        # Proceed with detailed scoring regardless of bug status
        logger.info("Proceeding with detailed scoring")

        # Use the scoring system
        for req in node.requirements:
            scoring_prompt: Prompt = {
                "Instructions": (
                    "You are an expert recommender system researcher reviewing code for an experiment."
                    "You are provided the research task, the code implementation and the execution output."
                    "Judge if the following requirement is fulfilled by the implementation. Be critical but fair."
                    "If the requirement is not fulfilled provide a short feedback of maximum a sentence on why it is not fulfilled and what needs to be changed to fulfill it."
                ),
                "Requirement": req.description,
                "Research Task": self.task_desc,
                "Implementation": node.code,
                "Execution output": node.term_out,
            }

            try:
                scoring_result = (
                    await Query()
                    .with_mcp(self._mcp_docs)
                    .with_system(
                        "Verify implementation against documented APIs when correctness is unclear. Reference usage documentation, prioritizing tutorials and user guides over source code."
                    )
                    .run(scoring_prompt, ScoreCode)
                )

                req.is_fulfilled = scoring_result.fulfilled
                req.feedback = scoring_result.feedback

            except Exception as e:
                logger.error(f"Error generate feedback for requirement: {req}")
                logger.error(f"Error in scoring: {e}")
                # Fallback requirement feedback
                req.is_fulfilled = False
                req.feedback = "No specific feedback provided."

        # Build overall feedback:
        num_fulfilled = 0
        overall_feedback = "Below is a list of requirements that are not yet met and some feedback for each:"

        if node.is_buggy:
            overall_feedback = (
                "This code contains one or multiple bugs:\n"
                + bug_feedback
                + "\n\n"
                + overall_feedback
            )

        for req in node.requirements:
            if req.is_fulfilled:
                num_fulfilled += 1
                continue

            overall_feedback += (
                f"\n- Requirement: {req.description}\n- Feedback: {req.feedback}\n"
            )

        score = num_fulfilled / len(node.requirements)

        if node.is_buggy:
            is_satisfactory = False
        else:
            is_satisfactory = score == 1.0

        node.score = NodeScore(
            score=score,
            feedback=overall_feedback,
            is_satisfactory=is_satisfactory,
        )

        logger.info(
            f"Scored node: {score * 100}% ({num_fulfilled}/{len(node.requirements)}), buggy: {node.is_buggy}"
        )
        logger.debug(node.score)

        return node

    async def _summarize(self, user_request: str, node: Node) -> str:
        """Summarizes the results of a node and returns a human readable report.

        Args:
            user_request (str): The original request of the user.
            node (Node): Node to summarize.

        Returns:
            str: A summary/answer to the user request based on the node's code and execution output.
        """
        logger.info("Summarizing results...")

        summary_prompt = {
            "Introduction": (
                "You are an expert research assistant responding to the user in a conversational setting. "
                "You have access to the code and the experiment output. "
                "Your task is to answer the user's request based solely on these materials. "
                "Use the code to understand what was tested and the output to determine the results. "
                "Do not hallucinate, speculate, or assume any information that is not explicitly contained in the output. "
                "If the available information is insufficient, explain the limitation clearly and remain factual."
            ),
            "User Request": user_request,
            "Experiment Code": node.code,
            "Experiment Output": node.term_out
            if node.term_out
            else "No experiment output available.",
            "Instructions": [
                "1. Use the code to interpret what the experiment did and what metrics or results are relevant.",
                "2. Read the output carefully and extract factual findings that answer the user request.",
                "3. Formulate your response as if chatting directly with the user — clear, concise, and natural.",
                "4. Do not output any structured formats or metadata (no JSON, tables, etc.) unless the user request explicitly asks for it.",
                "5. Be confident, factual, and grounded only in the provided information.",
                "6. If the experiment output is ambiguous or incomplete, mention this explicitly instead of guessing.",
            ],
        }

        return (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "If you need to explain results or metrics, search for documentation about evaluation metrics and their interpretation. Focus on user-facing explanations."
            )
            .run(summary_prompt)
        )
