from treesearch.backend.utils import FunctionSpec
from dataclasses import dataclass

select_datasets_spec = FunctionSpec(
    name="select_datasets",
    json_schema={
        "type": "object",
        "properties": {
            "selected_datasets": {
                "type": "array",
                "description": "List of dataset identifiers selected for the research task.",
                "items": {
                    "type": "string",
                    "description": "A dataset identifier from the available datasets list.",
                },
            }
        },
        "required": ["selected_datasets"],
    },
    description="Select appropriate datasets for the recommender system research task based on the task description and available datasets.",
)
@dataclass
class SelectDatasets:
    """Select appropriate datasets for the recommender system research task based on the task description and available datasets."""
    
    selected_datasets: list[str] # A List of dataset identifiers selected for the research task.

vlm_feedback_spec = FunctionSpec(
    name="analyze_experiment_plots",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the plot's results and implications",
                        },
                    },
                    "required": ["analysis"],
                },
            },
            "valid_plots_received": {
                "type": "boolean",
                "description": "True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.",
            },
            "vlm_feedback_summary": {
                "type": "string",
                "description": "Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.",
            },
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
    },
    description="Analyze experimental plots and provide detailed feedback on the results.",
)
@dataclass
class PlotAnalyses:
    """ Detailed analysis of the plot's results and implications """
    analysis: str # Detailed analysis of the plot's results and implications
    
@dataclass
class VLMFeedback:
    """ Analyze experimental plots and provide detailed feedback on the results. """
    plot_analyses : list[PlotAnalyses] # Detailed analysis of the plot's results and implications
    valid_plots_received : bool # True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.
    vlm_feedback_summary : str # Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.



review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

@dataclass
class ReviewFunction:
    """ Submit a review evaluating the output of the training script. """

    is_bug : bool # true if the output log shows that the execution failed or has some bug, otherwise false.
    summary : str # if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.


@dataclass
class ScoreCode:
    """Judge whether a single requirement is fulfilled by the code implementation and explain briefly."""
    
    fulfilled: bool # True if the specified requirement is fulfilled, false otherwise."""
    feedback: str # Short feedback explaining why the requirement is or isn't fulfilled."""

score_code_func_spec = FunctionSpec(
    name="score_code",
    json_schema={
        "type": "object",
        "properties": {
            "fulfilled": {
                "type": "boolean",
                "description": "True if the specified requirement is fulfilled, false otherwise.",
            },
            "feedback": {
                "type": "string",
                "description": "Short feedback explaining why the requirement is or isn't fulfilled.",
            },
        },
        "required": ["fulfilled", "feedback"],
    },
    description="Judge whether a single requirement is fulfilled by the code implementation and explain briefly.",
)

@dataclass
class RequirementJudgement:
    """ Judge whether a single requirement is fulfilled by the code implementation and explain briefly. """

    fulfilled : bool # True if the specified requirement is fulfilled, false otherwise.
    feedback : str # Short feedback explaining why the requirement is or isn't fulfilled.

set_code_requirements_spec = FunctionSpec(
    name="set_code_requirements",
    json_schema={
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "description": "A list of concise, clear and specific code requirements.",
                "items": {
                    "type": "string",
                    "description": "One specific requirement that must be met.",
                },
            }
        },
    },
    description=(
        "Set clear and specific code requirements for the implementation based on the research task."
    ),
)

@dataclass
class CodeRequirements:
        """Set clear and specific code requirements for the implementation based on the research task."""

        requirements: list[str]  # A list of concise, clear and specific code requirements.


plot_selection_spec = FunctionSpec(
    name="select_plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "description": "List of selected plot file paths",
                "items": {"type": "string", "description": "Full path to a plot file"},
                "maxItems": 10,
            }
        },
        "required": ["selected_plots"],
    },
    description="Select the 10 most relevant plots for analysis",
)

@dataclass
class PlotSelection:
    """Select the 10 most relevant plots for analysis"""

    selected_plots : list[str] # description": "List of selected plot file paths

    def __post_init__(self):
        if len(self.selected_plots) >10:
            raise ValueError(" list can not exceed 10 elements")
        

plan_and_code_spec = FunctionSpec(
    name="return_plan_and_code",
    json_schema={
        "type": "object",
        "properties": {
            "nl_text": {
                "type": "string",
                "description": "Explanatory natural language text describing the plan or reasoning behind the code.",
            },
            "code": {
                "type": "string",
                "description": "The complete Python source code implementing the plan.",
            },
        },
        "required": ["nl_text", "code"],
    },
    description="Return a natural language plan and the Python code that implements it.",
)

@dataclass
class PlanAndCode:
    """Return a natural language plan and the Python code that implements it.
    IMPORTANT: Do not use any markdown tags or similar in the code field. It MUST be plain and executable code.
    """

    nl_text : str # Explanatory natural language text describing the plan or reasoning behind the code.
    code : str # The complete plain and executable Python source code implementing the plan.