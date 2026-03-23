from pprint import pprint
from typing import Annotated, Any, Optional, Sequence, TypeAlias, TypedDict, TypeVar

from langchain.agents.structured_output import SchemaT
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

ResponseFormatType: TypeAlias = type[SchemaT]
RT = TypeVar("RT", bound=ResponseFormatType)


class AgentState[RT](TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_budget: int

    structured_response: Optional[RT]


class Agent:
    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        system_prompt: Optional[str] = None,
        response_schema: Optional[RT] = None,
        default_tool_budget: int = 25,
        tool_budget_warning: int = 5,
    ) -> None:
        self._system_prompt = system_prompt
        self._default_budget = default_tool_budget
        self._tool_budget_warning = tool_budget_warning

        self._response_tool_name = None
        all_tools: list[BaseTool | RT] = list(tools)
        bind_params = {}

        if response_schema is not None:
            self._response_tool_name = getattr(
                response_schema, "__name__", "FinalResponse"
            )
            all_tools.append(response_schema)
            bind_params["tool_choice"] = "required"

        pprint(f"{all_tools=}")
        pprint(f"{bind_params=}")

        self._model = model.bind_tools(all_tools, **bind_params)
        self._tool_executor = ToolNode(tools)

        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("llm_call", self._llm_call)
        graph.add_node("tools", self._tools_with_budget)
        graph.add_node("budget_enforcer", self._handle_budget_exceeded)

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call",
            self._after_llm_condition,
            {
                "tools": "tools",
                "force_stop": "budget_enforcer",
                "end": END,
            },
        )
        graph.add_edge("budget_enforcer", "llm_call")
        graph.add_edge("tools", "llm_call")

        return graph.compile()

    async def _llm_call(self, state: AgentState):
        tool_calls_left = state.get("tool_budget", self._default_budget)

        messages: list[BaseMessage] = []

        if self._system_prompt:
            messages.append(SystemMessage(self._system_prompt))

        if self._response_tool_name:
            messages.append(
                SystemMessage(
                    f"When you have the final answer, you MUST call the '{self._response_tool_name}' tool to respond."
                )
            )

        messages.extend(state["messages"])

        status_prompt = (
            f"[SYSTEM STATUS]\nCurrent Tool Budget: {tool_calls_left} calls remaining."
        )
        if tool_calls_left <= self._tool_budget_warning:
            status_prompt += "\nCRITICAL: You are almost out of tool calls. Do not start new broad searches. Use your remaining steps to synthesize an answer or perform one final targeted lookup."

        messages.append(SystemMessage(status_prompt))

        response = await self._model.ainvoke(messages)

        output: dict[str, Any] = {"messages": [response]}

        # If the LLM called the response_schema tool, extract arguments into state
        if isinstance(response, AIMessage) and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == self._response_tool_name:
                    output["structured_response"] = tool_call["args"]
                    break

        return output

    def _tools_with_budget(self, state: AgentState):
        result = self._tool_executor.invoke(state)
        return {**result, "tool_budget": state["tool_budget"] - 1}

    def _handle_budget_exceeded(self, state: AgentState):
        return {
            "messages": [
                SystemMessage(
                    "TERMINAL NOTICE: Tool execution blocked. Maximum budget reached. You MUST provide a final response to the user now based ONLY on existing information."
                )
            ]
        }

    def _after_llm_condition(self, state: AgentState):
        if state.get("structured_response") is not None:
            return "end"

        last_msg = state["messages"][-1]

        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            # If response tool was triggered, end graph
            for tool_call in last_msg.tool_calls:
                if tool_call["name"] == self._response_tool_name:
                    return "end"

            if state["tool_budget"] > 0:
                return "tools"

            return "force_stop"

        return "end"
