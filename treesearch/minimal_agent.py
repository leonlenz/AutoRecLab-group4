import json
import random
from pathlib import Path
from typing import Any, Optional

import humanize

from config import Config
from treesearch.backend.llm import query
from treesearch.function_specs import (
    plan_and_code_spec,
    review_func_spec,
    score_code_func_spec,
    set_code_requirements_spec,
)
from treesearch.interpreter import ExecutionResult
from treesearch.node import Node, NodeScore, Requirement
from treesearch.utils.response import wrap_code
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("nodeAgent")


class AblationIdea:
    """Ablation idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class HyperparamTuningIdea:
    """Hyperparameter tuning idea"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
    ):
        logger.info("Initializing agent...")
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        # HACK:
        # self.data_preview = None
        self._set_code_requirements()
        Path("./out/code_requirements.json").write_text(
            json.dumps(self.code_requirements)
        )
        logger.info("Agent initialized!")

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy==1.26.4",
            "numba==0.58.1",
            "pandas==2.3.2",
            "scipy==1.16.2",
            "scikit-learn==1.7.1",
            "lenskit==0.14.4",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!)."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "CRITICAL REQUIREMENTS - You should use lenskit as much as possible",
            "CRITICAL MODEL INPUT GUIDELINES:",
            "  - Always pay extra attention to the input to the model being properly normalized",
            "  - This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output",
        ]
        # HACK: Remove this for now
        # if hasattr(self.cfg.experiment, "num_syn_datasets"):
        #     num_syn_datasets = self.cfg.experiment.num_syn_datasets
        #     if num_syn_datasets > 1:
        #         impl_guideline.extend(
        #             [
        #                 f"You MUST evaluate your solution on at least {num_syn_datasets} different synthetic datasets to ensure robustness:",
        #                 "  - Use standard benchmark datasets when available",
        #                 f"  - If using synthetic data, generate at least {num_syn_datasets} variants with different characteristics",
        #                 "  - Report metrics separately for each dataset",
        #                 "  - Compute and report the average metric across all datasets",
        #             ]
        #         )
        impl_guideline.extend(
            [
                "For generative modeling tasks, you must:",
                "  - Generate a set of samples from your model",
                "  - Compare these samples with ground truth data using appropriate visualizations",
                "  - When saving plots, always use the 'working_dir' variable that will be defined at the start of the script",
                "  - Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.",
                "Important code structure requirements:",
                "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block",
                "  - All code should be at the global scope or in functions that are called from the global scope",
                "  - The script should execute immediately when run, without requiring any special entry point",
                "The code should start with:",
                "  import os",
                "  working_dir = os.path.join(os.getcwd(), 'working')",
                "  os.makedirs(working_dir, exist_ok=True)",
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Your response should only contain a single code block.",
                f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
                'You can also use the "./working" directory to store any temporary files that your code needs to create.',
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'dataset_name_1': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "      # Add additional datasets as needed:",
                "      'dataset_name_2': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "  }",
                "  # During training/evaluation:",
                "  experiment_data['dataset_name_1']['metrics']['train'].append(train_metric)",
                "  ```",
                "- Include timestamps or epochs with the saved metrics",
                "- For large datasets, consider saving in chunks or using np.savez_compressed()",
                "CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:",
                "  1. Track and print validation loss (if applicable) at each epoch or at suitable intervals:",
                "     ```python",
                "     print(f'Epoch {{epoch}}: validation_loss = {{val_loss:.4f}}')",
                "     ```",
                "  2. Track and update ALL these additional metrics: "
                + str(self.evaluation_metrics),
                "  3. Update metrics at EACH epoch:",
                "  4. Save ALL metrics at the end:",
                "     ```python",
                "     np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
                "     ```",
            ]
        )

        if self.cfg.agent.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.cfg.agent.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Make sure to write concise code."
            )
        }

    def _prompt_metricparse_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. "
            )
        }

    @property
    def _prompt_debug_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the bugfix/solution. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. Do not omit any part of the code, even if it was part of a previous implementation."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_hyperparam_tuning_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including hyperparameter tuning. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_ablation_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the ablation study. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an recommender systems researcher who is looking to publish a paper that will contribute significantly to the field."
                "Your first task is to write a python code to implement a solid baseline based on your research task provided below, "
                "from data preparation to model training, as well as evaluation and visualization. "
                "Focus on getting a simple but working implementation first, before any sophisticated improvements. "
                "We will explore more advanced variations in later stages."
            ),
            "Research task": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design. ",
                "The solution sketch should be 6-10 sentences. ",
                "Don't suggest to do EDA.",
                "Make sure to use the provided dataset(s).",
                "",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # if self.cfg.agent.data_preview:
        #     prompt["Data Overview"] = self.data_preview

        print("[cyan]--------------------------------[/cyan]")
        print("[cyan]self.task_desc[/cyan]")
        print("[cyan]" + self.task_desc + "[/cyan]")
        print("[cyan]--------------------------------[/cyan]")

        print("MinimalAgent: Getting plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("MinimalAgent: Draft complete")
        return self._new_node(plan, code)

    def _debug(self, parent_node: Node) -> Node:
        # Format node scores for the prompt
        score_info = ""
        if hasattr(parent_node, "score") and parent_node.score:
            score_info = f"""
                Previous Implementation Scores:
                - Score: {parent_node.score.score}%
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
                "You are an experienced recommender systems researcher. Your previous code for research experiment had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Research task": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Bug Analysis & Scoring": enhanced_bug_info + score_info,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_debug_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Pay special attention to the bug analysis and scoring feedback provided above.",
                "Address the specific errors or issues identified in the execution output.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        # if self.cfg.agent.data_preview:
        #     prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return self._new_node(plan, code, parent_node)

    def _improve(self, parent_node: Node) -> Node:
        # Format node scores for the prompt
        score_info = ""
        if hasattr(parent_node, "score") and parent_node.score:
            score_info = f"""
                Previous Implementation Scores:
                - Score: {parent_node.score.score}%
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
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Improvement guidelines": [
                "Based on the scoring feedback above, focus on the requirements that need improvement.",
                "Your goal is to enhance the implementation while maintaining its working functionality.",
            ]
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return self._new_node(plan, code, parent_node)

    def _new_node(self, plan: str, code: str, parent: Optional[Node] = None):
        return Node(
            plan=plan,
            code=code,
            _parent=parent,
            requirements=[Requirement(r) for r in self.code_requirements],
        )

    def _generate_seed_node(self, parent_node: Node):
        return Node(
            plan="Seed node",
            code=parent_node.code,
            _parent=parent_node,
            is_seed_node=True,
        )

    def _generate_hyperparam_tuning_node(
        self, parent_node: Node, hyperparam_idea: HyperparamTuningIdea
    ):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement hyperparameter tuning for the following idea: "
                + hyperparam_idea.name
                + ". "
                + hyperparam_idea.description
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'hyperparam_tuning_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "      },",
                "      # Add additional hyperparam tuning types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
            ]
        }
        prompt["Instructions"] |= self._prompt_hyperparam_tuning_resp_fmt
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan="Hyperparam tuning name: " + hyperparam_idea.name + ".\n" + plan,
            code=code,
            _parent=parent_node,
            hyperparam_name=hyperparam_idea.name,
        )

    def _generate_ablation_node(self, parent_node: Node, ablation_idea: AblationIdea):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement the ablation study for the following idea: "
                + ablation_idea.name
                + ". "
                + ablation_idea.description
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'ablation_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "          'dataset_name_2': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              # Add other relevant data",
                "          },",
                "      },",
                "      # Add additional ablation types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
            ]
        }
        prompt["Instructions"] |= self._prompt_ablation_resp_fmt
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan="Ablation name: " + ablation_idea.name + ".\n" + plan,
            code=code,
            _parent=parent_node,
            ablation_name=ablation_idea.name,
        )

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        # TODO: Refactor this to use the function spec
        # TODO: Integrate the code requirements in the plan. Must be removed in other functions like _debug and _improve since they call this function.
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            plan_and_code_result = query(
                system_message=prompt,
                user_message=None,
                func_spec=plan_and_code_spec,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.model_temp,
            )

            nl_text = plan_and_code_result.get("nl_text", "")
            code = plan_and_code_result.get("code", "")
            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _set_code_requirements(self):
        logger.info("Engineering code requirements...")
        requirements_prompt = f"""You are an expert recommender systems researcher. 
            You are provided with the following research task:\n{self.task_desc}\n" 
            Your job is to formulate a clear, concise list of essential requirements that the code implementation must fulfill to successfully address this research task. 
            Each requirement should be specific, actionable, and directly related to the research task. 
            Include all critical conceptual and technical requirements necessary for a successful experiment, but do not add unnecessary or generic requirements.
            A successful experiment means the code is technically AND conceptually correct, runs without errors, and produces meaningful results that align with the research task.
            Avoid vague language and keep each requirement as concise and precise as possible.
            """
        requirements_result = query(
            system_message=requirements_prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.model_temp,
            func_spec=set_code_requirements_spec,
        )
        self.code_requirements = requirements_result.get(
            "requirements", "No specific requirements provided."
        )
        
        # Requirements reflection round
        reflection_prompt = f"""You are an expert recommender systems researcher conducting a quality review.
            Your colleague generated code requirements that, when fulfilled, should result in a successful implementation of the research task.

            This is the research task: {self.task_desc}

            Your colleague generated the following code requirements:
            {self.code_requirements}
            
            Review these requirements and verify they meet ALL these criteria:
            1. Each requirement is specific, actionable, and measurable
            2. All requirements are directly relevant to the research task
            3. ALL critical technical AND conceptual aspects are covered
            4. No vague, generic, or redundant requirements are included
            5. Requirements focus on successful experiment execution and meaningful results
            
            Provide an updated, refined list of requirements that fixes any issues found. Keep requirements that already meet the criteria unchanged.
            """
        reflection_result = query(
            system_message=reflection_prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.model_temp,
            func_spec=set_code_requirements_spec,
        )
        self.code_requirements = reflection_result.get(
            "requirements", "No specific requirements provided."
        )
        logger.info("Done.")

    def score_code(self, node: Node, exec_result: ExecutionResult) -> Node:
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
            "Implementation": wrap_code(node.code),
            "Execution Output": wrap_code(
                node.term_out if node.term_out else "No output generated", lang=""
            ),
            "Instructions": [
                "Carefully analyze the execution output for signs of bugs or failures:",
                "- Syntax errors, import errors, or runtime exceptions",
                "- Missing required outputs or metrics",
                "- Execution timeouts or crashes",
                "- Incorrect or nonsensical results",
                "",
                "If there's a bug, provide a clear summary of the issue and suggest how to fix it.",
                "If the execution was successful, leave the summary empty.",
            ],
        }

        try:
            review_result = query(
                system_message=review_prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.model_temp,
            )

            # Update node with review results
            node.is_buggy = review_result.get("is_bug", True)
            node.analysis = review_result.get("summary", "")

            if node.is_buggy:
                logger.info(f"Node identified as buggy: {node.analysis}")
                # Create more helpful feedback for buggy nodes
                helpful_feedback = f"""EXECUTION FAILURE DETECTED:

                    {node.analysis}

                    NEXT STEPS FOR DEBUGGING:
                    - Review the error message above carefully
                    - Check for missing imports or incorrect package names
                    - Verify variable names and function calls
                    - Ensure all required data files are accessible
                    - Consider simplifying the code to isolate the issue

                    This implementation scored 0% due to execution failure. Focus on resolving the error before optimizing."""

                # If buggy, set minimal scores and return early
                node.score = NodeScore(
                    score=0.0,
                    feedback=helpful_feedback,
                    is_satisfactory=False,
                )
                return node

        except Exception as e:
            logger.error(f"Error in code review: {e}")
            # Fallback: mark as buggy if analysis fails
            node.is_buggy = True
            node.analysis = f"Review analysis failed: {str(e)}"

            fallback_feedback = f"""ANALYSIS SYSTEM ERROR:

                The automated review system encountered an error: {str(e)}

                MANUAL REVIEW REQUIRED:
                - Check the execution output manually for obvious errors
                - Look for common issues like import errors, syntax errors, or missing dependencies
                - Verify that all required packages are installed
                - Test the code in smaller chunks to isolate any problems

                This implementation scored 0% due to analysis failure. Manual debugging recommended."""

            node.score = NodeScore(
                score=0.0,
                feedback=fallback_feedback,
                is_satisfactory=False,
            )
            return node

        # If not buggy, proceed with detailed scoring
        logger.info("Node execution successful, proceeding with detailed scoring")

        # If the node is not buggy, use the scoring system
        scoring_prompt = {
            f"Instructions": (
                f"""You are an expert recommender system researcher conducting a code review for an important experiment. You are provided the research task, the code implementation and the execution output. You must score the code using the following requirements: \n{self.code_requirements}\n. Each requirement can either be fullfilled or not fulfilled. The score is the percentage of requirements that are fulfilled, e.g. if 7 out of 10 requirements are fulfilled, the score is 70. Additionally, you provide clear and constructive feedback how to improve the code implementation based on the requirements."""
            ),
            f"Research Task": (self.task_desc),
            f"Implementation": (wrap_code(node.code)),
            f"Execution output": (wrap_code(node.term_out, lang="")),
        }

        try:
            scoring_result = query(
                system_message=scoring_prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.model_temp,
                func_spec=score_code_func_spec,
            )

            # Parse the structured response
            node.score = NodeScore(
                score=scoring_result.get("score", 0.0),
                feedback=scoring_result.get("feedback", ""),
                is_satisfactory=scoring_result.get("is_satisfactory", False),
            )

        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            # Fallback scoring for successful but unscored code
            node.score = NodeScore(
                score=0.0,
                feedback=f"Scoring failed but code executed successfully: {str(e)}",
                is_satisfactory=False,
            )

        print(node.score)

        return node

        # # Create scoring criteria summary
        # criteria_summary = "\n".join(
        #     [
        #         f"**{cat['name']}** (0-10): {cat['description']}\n"
        #         + "\n".join([f"  - {criterion}" for criterion in cat["criteria"]])
        #         for cat in scoring_categories.values()
        #     ]
        # )

        # scoring_prompt = {
        #     "Introduction": (
        #         "You are an expert Recommender System researcher conducting a paper review. "
        #         "Your task is to evaluate code implementation using a structured scoring system (0-10 scale). "
        #         "A score of 0 means complete failure, and 10 means excellent implementation. "
        #         "Provide objective, constructive feedback."
        #     ),
        #     "Research Task": self.task_desc,
        #     # HACK:
        #     # "Required Metrics": f"Must evaluate: {', '.join(self.evaluation_metrics)}",
        #     "Implementation": wrap_code(node.code),
        #     "Execution Output": wrap_code(node.term_out, lang=""),
        #     "Scoring Categories": criteria_summary,
        #     "Scoring Instructions": [
        #         "Score each category from 0-10:",
        #         "- 0: Complete failure or critical errors",
        #         "- 1-3: Major issues preventing success",
        #         "- 4-6: Partial implementation with significant problems",
        #         "- 7-8: Good implementation with minor issues",
        #         "- 9-10: Excellent implementation",
        #         "",
        #         "Since the code executed successfully, focus on quality and correctness.",
        #         "",
        #         "RESPONSE FORMAT - You MUST structure your response as follows:",
        #         "EXPERIMENT_ACHIEVEMENT: [score 0-10]",
        #         "CODE_QUALITY: [score 0-10]",
        #         "CONCEPTUAL_CORRECTNESS: [score 0-10]",
        #         "OVERALL_SCORE: [average of above three scores]",
        #         "FEEDBACK: [Short feedback with specific improvement suggestions]",
        #     ],
        # }

    def _summarize(self, user_request: str, node: Node) -> str:
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
            "Experiment Code": wrap_code(node.code),
            "Experiment Output": wrap_code(
                node.term_out if node.term_out else "No experiment output available.",
                lang="",
            ),
            "Instructions": [
                "1. Use the code to interpret what the experiment did and what metrics or results are relevant.",
                "2. Read the output carefully and extract factual findings that answer the user request.",
                "3. Formulate your response as if chatting directly with the user — clear, concise, and natural.",
                "4. Do not output any structured formats or metadata (no JSON, tables, etc.) unless the user request explicitly asks for it.",
                "5. Be confident, factual, and grounded only in the provided information.",
                "6. If the experiment output is ambiguous or incomplete, mention this explicitly instead of guessing.",
            ],
        }

        # HACK: Str casting:
        # We should make query generic or split it into 2 function for normal query and for tool usage or smth
        return str(
            query(
                summary_prompt,
                None,
                self.cfg.agent.code.model,
                self.cfg.agent.code.model_temp,
            )
        )

    def _generate_plotting_code(
        self,
        node: Node,
        working_dir: str,
        plot_code_from_prev_stage: Optional[str] = None,
    ) -> str:
        """Generate code for plotting experiment results"""
        prompt_guideline = [
            "AVAILABLE DATA: ",
            "Experiment Data: experiment_data.npy",
        ]
        prompt_guideline += [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Use the following experiment code to infer the data to plot: " + node.code,
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
        ]
        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
                except Exception as e:
                    print(f'Error loading experiment data: {{e}}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {{e}}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {{e}}")
                    plt.close()
            """,
        ]
        # add instruction for format
        plotting_prompt = {
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= self._prompt_resp_fmt
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }

        # For stage 3, initialize with stage 2's plotting code
        if (
            self.stage_name
            and self.stage_name.startswith("3_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: Use the following base plotting code as a starting point:",
                    "Base plotting code: " + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between different datasets",
                    "3. Add dataset-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from which dataset",
                    "5. Use consistent naming conventions for saved files",
                ]
            )
        # For stage 4, initialize with stage 3's plotting code
        elif (
            self.stage_name
            and self.stage_name.startswith("4_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
                    "Base plotting code: \n" + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between ablation and baseline results",
                    "3. Add ablation-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from ablation vs baseline",
                    "5. Use consistent naming conventions for saved files",
                ]
            )

        # Get plotting code from LLM
        plan, code = self.plan_and_code_query(plotting_prompt)

        # Ensure the code starts with imports
        if not code.strip().startswith("import"):
            code = "import matplotlib.pyplot as plt\nimport numpy as np\n\n" + code

        node.plot_code = code
        node.plot_plan = plan

        return code

    # def _determine_datasets_successfully_tested(self, node: Node) -> list[str]:
    #     """Determine which datasets are successfully tested based on VLM feedback"""
    #     plot_analyses = ""
    #     for i, plot_analysis in enumerate(node.plot_analyses):
    #         # FIXME: Type error:
    #         plot_analyses += f"plot {i + 1}: {plot_analysis['analysis']}\n"
    # HACK: Comment this for now
    # def _determine_datasets_successfully_tested(self, node: Node) -> list[str]:
    #     """Determine which datasets are successfully tested based on VLM feedback"""
    #     plot_analyses = ""
    #     for i, plot_analysis in enumerate(node.plot_analyses):
    #         # FIXME: Type error:
    #         plot_analyses += f"plot {i + 1}: {plot_analysis['analysis']}\n"

    #     determine_prompt = {
    #         "Introduction": "You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed.",
    #         "Plot analyses": plot_analyses,
    #         "VLM feedback summary": node.vlm_feedback_summary,
    #         "Original plotting code": node.plot_code,
    #         "Response format": (
    #             "Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line."
    #             "In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: <list_datasets_successfully_tested>', "
    #         ),
    #     }
    #     determine_prompt = {
    #         "Introduction": "You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed.",
    #         "Plot analyses": plot_analyses,
    #         "VLM feedback summary": node.vlm_feedback_summary,
    #         "Original plotting code": node.plot_code,
    #         "Response format": (
    #             "Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line."
    #             "In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: <list_datasets_successfully_tested>', "
    #         ),
    #     }

    #     retry_count = 0
    #     retry_limit = 5
    #     while retry_count < retry_limit:
    #         response = query(
    #             system_message=determine_prompt,
    #             user_message=None,
    #             model=self.cfg.agent.feedback.model,
    #             temperature=self.cfg.agent.feedback.temp,
    #         )
    #     retry_count = 0
    #     retry_limit = 5
    #     while retry_count < retry_limit:
    #         response = query(
    #             system_message=determine_prompt,
    #             user_message=None,
    #             model=self.cfg.agent.feedback.model,
    #             temperature=self.cfg.agent.feedback.temp,
    #         )

    #         (
    #             reasoning,
    #             datasets_successfully_tested_str,
    #         ) = self._parse_keyword_prefix_response(
    #             response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
    #         )
    #         print(f"[green]Reasoning:[/green] {reasoning}")
    #         print(
    #             f"[green]Datasets successfully tested:[/green] {datasets_successfully_tested_str}"
    #         )
    #         if reasoning is not None and datasets_successfully_tested_str is not None:
    #             if datasets_successfully_tested_str == "":
    #                 return [""]
    #             # Split by comma and clean each dataset name
    #             datasets = [
    #                 ds.strip() for ds in datasets_successfully_tested_str.split(",")
    #             ]
    #             # Filter out empty strings and ensure all elements are strings
    #             datasets = [ds for ds in datasets if isinstance(ds, str) and ds]
    #             logger.info(f"Successfully parsed datasets: {datasets}")
    #             return datasets
    #         (
    #             reasoning,
    #             datasets_successfully_tested_str,
    #         ) = _parse_keyword_prefix_response(
    #             response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
    #         )
    #         print(f"[green]Reasoning:[/green] {reasoning}")
    #         print(
    #             f"[green]Datasets successfully tested:[/green] {datasets_successfully_tested_str}"
    #         )
    #         if reasoning is not None and datasets_successfully_tested_str is not None:
    #             if datasets_successfully_tested_str == "":
    #                 return [""]
    #             # Split by comma and clean each dataset name
    #             datasets = [
    #                 ds.strip() for ds in datasets_successfully_tested_str.split(",")
    #             ]
    #             # Filter out empty strings and ensure all elements are strings
    #             datasets = [ds for ds in datasets if isinstance(ds, str) and ds]
    #             logger.info(f"Successfully parsed datasets: {datasets}")
    #             return datasets

    #         retry_count += 1
    #         logger.warning(
    #             f"Failed to parse successfully tested datasets response (attempt {retry_count}/{retry_limit})"
    #         )
    #         retry_count += 1
    #         logger.warning(
    #             f"Failed to parse successfully tested datasets response (attempt {retry_count}/{retry_limit})"
    #         )

    #     logger.error(
    #         f"Failed to parse successfully tested datasets response after {retry_limit} retries. Falling back to an empty list."
    #     )
    #     return [""]
    #     logger.error(
    #         f"Failed to parse successfully tested datasets response after {retry_limit} retries. Falling back to an empty list."
    #     )
    #     return [""]

    # def _analyze_plots_with_vlm(self, node: Node) -> None:
    #     """Analyze experimental plots using VLM"""
    #     if not node.plot_paths:
    #         return
    # HACK: Comment this for now
    # def _analyze_plots_with_vlm(self, node: Node) -> None:
    #     """Analyze experimental plots using VLM"""
    #     if not node.plot_paths:
    #         return

    #     # for debugging
    #     print(f"[cyan]Plot paths:[/cyan] {node.plot_paths}")
    #     # for debugging
    #     print(f"[cyan]Plot paths:[/cyan] {node.plot_paths}")

    #     def encode_image_to_base64(image_path):
    #         with open(image_path, "rb") as image_file:
    #             try:
    #                 return base64.b64encode(image_file.read()).decode("utf-8")
    #             except Exception as e:
    #                 print(f"[red]Error encoding image {image_path}: {e}[/red]")
    #                 return None
    #     def encode_image_to_base64(image_path):
    #         with open(image_path, "rb") as image_file:
    #             try:
    #                 return base64.b64encode(image_file.read()).decode("utf-8")
    #             except Exception as e:
    #                 print(f"[red]Error encoding image {image_path}: {e}[/red]")
    #                 return None

    #     if not len(node.plot_paths) > 10:
    #         selected_plots = node.plot_paths
    #     else:
    #         print(
    #             f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. Calling LLM to select the most relevant plots to analyze.[/red]"
    #         )
    #         # select 10 plots to analyze
    #         prompt_select_plots = {
    #             "Introduction": (
    #                 "You are an experienced AI researcher analyzing experimental results. "
    #                 "You have been provided with plots from a machine learning experiment. "
    #                 "Please select 10 most relevant plots to analyze. "
    #                 "For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs."
    #                 "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
    #             ),
    #             "Plot paths": node.plot_paths,
    #         }
    #     if not len(node.plot_paths) > 10:
    #         selected_plots = node.plot_paths
    #     else:
    #         print(
    #             f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. Calling LLM to select the most relevant plots to analyze.[/red]"
    #         )
    #         # select 10 plots to analyze
    #         prompt_select_plots = {
    #             "Introduction": (
    #                 "You are an experienced AI researcher analyzing experimental results. "
    #                 "You have been provided with plots from a machine learning experiment. "
    #                 "Please select 10 most relevant plots to analyze. "
    #                 "For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs."
    #                 "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
    #             ),
    #             "Plot paths": node.plot_paths,
    #         }

    #         try:
    #             response_select_plots = cast(
    #                 dict,
    #                 query(
    #                     system_message=prompt_select_plots,
    #                     user_message=None,
    #                     func_spec=plot_selection_spec,
    #                     model=self.cfg.agent.feedback.model,
    #                     temperature=self.cfg.agent.feedback.temp,
    #                 ),
    #             )
    #         try:
    #             response_select_plots = cast(
    #                 dict,
    #                 query(
    #                     system_message=prompt_select_plots,
    #                     user_message=None,
    #                     func_spec=plot_selection_spec,
    #                     model=self.cfg.agent.feedback.model,
    #                     temperature=self.cfg.agent.feedback.temp,
    #                 ),
    #             )

    #             print(f"[cyan]Plot selection response:[/cyan] {response_select_plots}")
    #             # Extract the plot paths list
    #             selected_plots = response_select_plots.get("selected_plots", [])
    #             print(f"[cyan]Plot selection response:[/cyan] {response_select_plots}")
    #             # Extract the plot paths list
    #             selected_plots = response_select_plots.get("selected_plots", [])

    #             # Validate that all paths exist and are image files
    #             valid_plots = []
    #             for plot_path in selected_plots:
    #                 if (
    #                     isinstance(plot_path, str)
    #                     and os.path.exists(plot_path)
    #                     and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
    #                 ):
    #                     valid_plots.append(plot_path)
    #                 else:
    #                     logger.warning(f"Invalid plot path received: {plot_path}")
    #             # Validate that all paths exist and are image files
    #             valid_plots = []
    #             for plot_path in selected_plots:
    #                 if (
    #                     isinstance(plot_path, str)
    #                     and os.path.exists(plot_path)
    #                     and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
    #                 ):
    #                     valid_plots.append(plot_path)
    #                 else:
    #                     logger.warning(f"Invalid plot path received: {plot_path}")

    #             # Use the validated list
    #             if valid_plots:
    #                 print(f"[cyan]Selected valid plots:[/cyan] {valid_plots}")
    #                 selected_plots = valid_plots
    #             else:
    #                 logger.warning(
    #                     "No valid plot paths found in response, falling back to first 10 plots"
    #                 )
    #                 # fallback to first 10 plots
    #                 # validate node.plot_paths
    #                 selected_plots = []
    #                 for plot_path in node.plot_paths[:10]:
    #                     if os.path.exists(plot_path) and plot_path.lower().endswith(
    #                         (".png", ".jpg", ".jpeg")
    #                     ):
    #                         selected_plots.append(plot_path)
    #                     else:
    #                         logger.warning(f"Invalid plot path received: {plot_path}")
    #             # Use the validated list
    #             if valid_plots:
    #                 print(f"[cyan]Selected valid plots:[/cyan] {valid_plots}")
    #                 selected_plots = valid_plots
    #             else:
    #                 logger.warning(
    #                     "No valid plot paths found in response, falling back to first 10 plots"
    #                 )
    #                 # fallback to first 10 plots
    #                 # validate node.plot_paths
    #                 selected_plots = []
    #                 for plot_path in node.plot_paths[:10]:
    #                     if os.path.exists(plot_path) and plot_path.lower().endswith(
    #                         (".png", ".jpg", ".jpeg")
    #                     ):
    #                         selected_plots.append(plot_path)
    #                     else:
    #                         logger.warning(f"Invalid plot path received: {plot_path}")

    #         except Exception as e:
    #             logger.error(
    #                 f"Error in plot selection: {str(e)}; falling back to first 10 plots"
    #             )
    #             # Fallback to using first 10 plots
    #             selected_plots = node.plot_paths[:10]
    #         except Exception as e:
    #             logger.error(
    #                 f"Error in plot selection: {str(e)}; falling back to first 10 plots"
    #             )
    #             # Fallback to using first 10 plots
    #             selected_plots = node.plot_paths[:10]

    #     print("[cyan]Before encoding images[/cyan]")
    #     user_message = [
    #         {
    #             "type": "text",
    #             "text": (
    #                 "You are an experienced AI researcher analyzing experimental results. "
    #                 "You have been provided with plots from a machine learning experiment. "
    #                 f"This experiment is based on the following research task: {self.task_desc}"
    #                 "Please analyze these plots and provide detailed insights about the results. "
    #                 "If you don't receive any plots, say 'No plots received'. "
    #                 "Never make up plot analysis. "
    #                 "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
    #                 "like 'the first plot'."
    #             ),
    #         }
    #     ] + [
    #         {
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{encode_image_to_base64(plot_path)}"
    #             },
    #         }
    #         for plot_path in selected_plots
    #     ]
    #     print("[cyan]Before encoding images[/cyan]")
    #     user_message = [
    #         {
    #             "type": "text",
    #             "text": (
    #                 "You are an experienced AI researcher analyzing experimental results. "
    #                 "You have been provided with plots from a machine learning experiment. "
    #                 f"This experiment is based on the following research task: {self.task_desc}"
    #                 "Please analyze these plots and provide detailed insights about the results. "
    #                 "If you don't receive any plots, say 'No plots received'. "
    #                 "Never make up plot analysis. "
    #                 "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
    #                 "like 'the first plot'."
    #             ),
    #         }
    #     ] + [
    #         {
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{encode_image_to_base64(plot_path)}"
    #             },
    #         }
    #         for plot_path in selected_plots
    #     ]

    #     response = cast(
    #         dict,
    #         query(
    #             system_message=None,
    #             user_message=user_message,
    #             func_spec=vlm_feedback_spec,
    #             model=self.cfg.agent.vlm_feedback.model,
    #             temperature=self.cfg.agent.vlm_feedback.temp,
    #         ),
    #     )
    #     print(
    #         f"[cyan]VLM response from {self.cfg.agent.vlm_feedback.model}:[/cyan] {response}"
    #     )
    #     if response["valid_plots_received"]:
    #         node.is_buggy_plots = False
    #     else:
    #         node.is_buggy_plots = True
    #     response = cast(
    #         dict,
    #         query(
    #             system_message=None,
    #             user_message=user_message,
    #             func_spec=vlm_feedback_spec,
    #             model=self.cfg.agent.vlm_feedback.model,
    #             temperature=self.cfg.agent.vlm_feedback.temp,
    #         ),
    #     )
    #     print(
    #         f"[cyan]VLM response from {self.cfg.agent.vlm_feedback.model}:[/cyan] {response}"
    #     )
    #     if response["valid_plots_received"]:
    #         node.is_buggy_plots = False
    #     else:
    #         node.is_buggy_plots = True

    #     for index, analysis in enumerate(response["plot_analyses"]):
    #         analysis["plot_path"] = node.plot_paths[index]
    #     for index, analysis in enumerate(response["plot_analyses"]):
    #         analysis["plot_path"] = node.plot_paths[index]

    #     node.plot_analyses = response["plot_analyses"]
    #     node.vlm_feedback_summary = response["vlm_feedback_summary"]
    #     node.plot_analyses = response["plot_analyses"]
    #     node.vlm_feedback_summary = response["vlm_feedback_summary"]

    #     node.datasets_successfully_tested = (
    #         self._determine_datasets_successfully_tested(node)
    #     )
    #     node.datasets_successfully_tested = (
    #         self._determine_datasets_successfully_tested(node)
    #     )

    # def _generate_node_summary(self, node: Node) -> dict:
    #     """Generate a summary of the node's experimental findings"""
    #     summary_prompt = {
    #         "Introduction": (
    #             "You are a recommender system researcher analyzing experimental results. "
    #             "Please summarize the findings from this experiment iteration."
    #         ),
    #         "Research task": self.task_desc,
    #         "Implementation": wrap_code(node.code),
    #         "Plan": node.plan,
    #         "Execution output": wrap_code(node.term_out, lang=""),
    #         "Analysis": node.analysis,
    #         "Metric": str(node.metric) if node.metric else "Failed",
    #         "Plot Analyses": (
    #             node.plot_analyses if hasattr(node, "plot_analyses") else []
    #         ),
    #         "VLM Feedback": (
    #             node.vlm_feedback_summary
    #             if hasattr(node, "vlm_feedback_summary")
    #             else ""
    #         ),
    #     }

    #     return cast(
    #         dict,
    #         query(
    #             system_message=summary_prompt,
    #             user_message=None,
    #             # FIXME: Type error
    #             func_spec={
    #                 "name": "summarize_experiment",
    #                 "description": "Summarize experimental findings",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "findings": {
    #                             "type": "string",
    #                             "description": "Key findings and results",
    #                         },
    #                         "significance": {
    #                             "type": "string",
    #                             "description": "Why these results matter",
    #                         },
    #                         "next_steps": {
    #                             "type": "string",
    #                             "description": "Suggested improvements or next experiments",
    #                         },
    #                     },
    #                     "required": ["findings", "significance"],
    #                 },
    #             },
    #             model=self.cfg.agent.feedback.model,
    #             temperature=self.cfg.agent.feedback.temp,
    #         ),
    #     )
