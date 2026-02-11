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
            f"0. Documentation Protocol: OmniRec is the primary framework. Before writing a single line of code, you MUST use the documentation tool to search for the specific classes and methods you plan to use. Do not guess parameters. You have access to the documentation of these libraries: {VECTOR_STORE_NAMES}",
            "1. Python Framework - !CRITICAL!: You have access to the OmniRec python library, a new comprehensive recommender system framework. Within this framework you can use algorithms from Lenskit, RecBole, RecPack and Elliot. If you use algorithms from these libraries, you MUST use OmniRec.",
            f"2. Datasets: Use only the following selected datasets for training and evaluation: {self.selected_datasets}",
            "3. Code Structure:",
            "   - Single-file, self-contained Python script.",
            "   - ALWAYS wrap the program’s starting point in `if __name__ == '__main__':` so it only runs when the script is executed directly.",
            "   - All code at global scope or in functions called from global scope.",
            "   - Keep it SIMPLE: Use only essential, well-documented APIs. Avoid unnecessary operations and complexity.",
            "5. Environment & Output:",
            "   - Start with:",
            "     import os",
            "     working_dir = os.path.join(os.getcwd(), 'working')",
            "     os.makedirs(working_dir, exist_ok=True)",
            f"   - Ensure execution completes within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            "6. Data Saving:",
            "   - Save all metrics, losses, and predictions in a dictionary `experiment_data`.",
            "   - Save this dictionary at the end: `np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)`.",
            "7. Evaluation - Track these throughout execution and use for final evaluation:",
            f"   - Metrics: {', '.join(self.evaluation_metrics) if self.evaluation_metrics else 'Use metrics from task description if specified, otherwise choose appropriate ones'}",
            "   - Print metrics during training for monitoring if practical.",
            "8. CRITICAL API USAGE - Verify in documentation to avoid common errors:",
            "   - Verify object attributes exist before accessing (e.g., check SplitData structure)",
            "   - Always check constructor signatures - don't assume parameters exist",
            "   - Use public attributes/methods, NOT private ones starting with underscore",
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
                "2. DOCUMENTATION VERIFICATION: If the error involves OmniRec, Lenskit, or RecBole, you MUST search the documentation for the correct class/method signature before writing the fix.",
                "3. EXPLAIN THE FIX: Write 3-5 sentences describing the root cause and the verified solution. Cite the documentation if an API change was made.",
                "4. DO NOT GUESS: If the documentation does not show the method you need, search for alternatives or 'tutorials' in the MCP server.",
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
                "implementation. Your task is to improve it based on the current experimental stage."
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
                "You are a Senior Recommender Systems Engineer specializing in the OmniRec library.\n"
                "You work in a strict 'Test-Driven' and 'Doc-Driven' environment.\n"
                "### OPERATIONAL CONSTRAINTS:\n"
                "1. MANDATORY SEARCH: You are prohibited from generating code until you have performed at least two search queries.\n"
                "- Query 1: Broad search for the class/tutorial.\n"
                "- Query 2: Specific search for method signatures and parameter types.\n"
                "2. CITATION RULE: In your 'nl_text' field, you MUST start with a section titled '## Documentation Verified'. List every OmniRec method you used and the parameters you confirmed via the search tool.\n"
                "3. THE 'NO-GUESS' ARCHITECTURE: If the documentation tool does not return a specific parameter you need, do not 'hallucinate' it. Instead, search for 'OmniRec [ClassName] examples' to see it in context.\n"
                "4. VERSION AWARENESS: You are using OmniRec v0.2.0. Disregard pre-trained knowledge of Lenskit or RecBole if it conflicts with the current OmniRec documentation retrieved via MCP.\n"
                "### THE THREE-PHASE PROTOCOL:\n"
                "Phase 1 (Identify): Analyze the task and list the 3-5 key OmniRec components needed.\n"
                "Phase 2 (Verify): For EACH component, call the MCP tool. If the search result is a list of methods, you MUST perform a follow-up search on the specific method signature/\n"
                "Phase 3 (Implement): Only when you can 'see' the documentation in your context window are you allowed to populate the PlanAndCode fields."
            )
            .run(prompt, PlanAndCode)
        )

        nl_text = plan_and_code_result.nl_text
        code = strip_markdown_fences(plan_and_code_result.code)
        return nl_text, code
    
    # Alternative, shorter system prompt:
    """
        f"CRITICAL: Before writing code, search the respective documentation. You have access to comprehensive documentation for these libraries: {VECTOR_STORE_NAMES}. Always verify the following technical details in the documentation before using them in your code: "
        "1) Exact function signatures (parameter names, types, valid value ranges), "
        "2) Object attributes (use public APIs, not private _attributes), "
        "3) Data structures."
        "Never guess - always verify in docs."
    """

    async def _select_datasets(self) -> list[str]:
        """Select appropriate datasets for the research task using LLM."""
        prompt: Prompt = {
            "Instruction:": (
                f"You are a recommender system researcher who wants to implement a given research task. "
                f"Your first task is to select suitable datasets for the task. "
                f"Please first look at the research task and see if it specifies any datasets:\n{self.task_desc}\n"
                "Select the identifiers of the specified datasets or if none are specified choose appropriate datasets from the list of available datasets below."
                "Your response MUST just be a simple list of dataset identifiers."
                f"Here are all available datasets with their identifiers and brief statistics:\n{get_datasets_table()}"
            )
        }
        result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "If you need information about dataset characteristics or recommender system domains, search the OmniRec documentation for dataset usage."
            )
            .run(prompt, SelectDatasets)
        )
        return result.selected_datasets

    async def _set_code_requirements(self):
        logger.info("Engineering code requirements...")
        requirements_prompt = f"""
        ROLE:
        You are an expert recommender systems researcher with extensive experience in designing and implementing experiments to advance the field.

        CONTEXT:
        You are provided with the following research task:
        {self.task_desc}
        And here are the selected datasets for this task:
        {self.selected_datasets}

        GOAL:
        Formulate a clear, concise list of essential requirements that the code implementation must fulfill to successfully address this research task.

        CRITICAL REQUIREMENT GUIDELINES:
        1. Function over Form: You MUST only focus on whats really necessary for the experiment to work.
        2. Specificity: Each requirement must be actionable and directly related to the research task.
        3. Scope: Requirements should be specific enough for the task but broad enough to allow for valid implementation variations.
        4. Atomicity: Requirements MUST NOT include any sub-requirements and must be atomic.
        5. Coverage: Include all critical conceptual (!IMPORTANT!) and technical requirements necessary for a successful experiment. DO NOT add unnecessary requirements.
        6. Success Criteria: A successful experiment means the code is technically AND conceptually correct and follows best practices, runs without errors, and produces meaningful results that align with the research task. The data splitting, algorithm configuration and evaluation MUST BE suitable for the provided data (explicit or implicit) and the research task.
        7. Style: Avoid vague and verbose language. Keep each requirement as concise and precise as possible.
        """
        requirements_result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "Search documentation technical details of the OmniRec framework and selected datasets to ensure requirements are feasible. Prioritize implementation guides and API references."
            )
            .run(requirements_prompt, CodeRequirements)
        )
        if len(requirements_result.requirements) == 0:
            self.code_requirements = "No specific requirements provided."
        else:
            self.code_requirements = requirements_result.requirements

        # Requirements reflection round
        reflection_prompt = f"""
        You are an expert recommender systems researcher conducting a quality review.
        Your colleague generated code requirements that, when fulfilled, should result in a successful implementation of the research task.
        This is the research task:
        {self.task_desc}

        And here are the generated requirements:
        {self.code_requirements}

        GOAL:
        Review these requirements critically but fairly and provide an updated, refined list.

        GUIDELINES:
        1. Verification: Verify that the requirements meet ALL these criteria:
           - Specificity & Atomicity: Each requirement MUST BE specific, actionable, and atomic (no sub-requirements).
           - Relevance & Scope: Requirements MUST BE directly relevant but broad enough to allow for valid implementation variations (avoid over-specificity).
           - Coverage: ALL critical technical AND conceptual aspects (!IMPORTANT!) are covered, including data splitting, algorithm configuration, and evaluation suitability for the provided data (explicit or implicit) and research task.
           - Clarity: No vague, generic, redundant or unecessarily strict requirements are included.
           - Focus: Requirements focus on successful experiment execution and meaningful results.
        2. Refinement: Fix any issues found. Keep requirements that already meet the criteria unchanged.
        """
        reflection_result = (
            await Query()
            .with_mcp(self._mcp_docs)
            .with_system(
                "Verify requirements against documented best practices. Search the documentation to make sure that the technical details of the requirements are correct."
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
                    "When diagnosing bugs, search for usage examples in documentation. Look for common error patterns and correct API usage."
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
                        "Check implementation against documented APIs and examples. Search for usage documentation to verify correctness, prioritizing tutorials and user guides over source code."
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
