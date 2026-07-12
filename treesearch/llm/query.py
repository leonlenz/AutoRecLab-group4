import asyncio
import sys
from dataclasses import dataclass
from typing import Optional, Self, TypeAlias, TypeVar, overload

from langchain.agents.structured_output import SchemaT
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_openai import ChatOpenAI

from config import get_config, is_reasoning_model
from treesearch.llm.graph import Agent
from treesearch.utils.costs_tracker import TokenUsageOpenAi, get_cost_tracker
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("llm")
tracker = get_cost_tracker()

# Sentinel so callers can explicitly pass reasoning_effort=None ("omit") and have
# it honoured, while a missing argument still falls back to the config default.
_UNSET = object()

# Per-request timeout (seconds) for a single OpenAI call. Combined with
# max_retries, a stalled request fails fast and is retried instead of hanging.
_REQUEST_TIMEOUT_S = 300
# Hard ceiling (seconds) for one full agent run (the whole tool-calling loop).
# Backstop against any await hanging forever — a stalled OpenAI/MCP request
# once wedged a run for ~30h. Generous vs. a normal loop (a few minutes).
_AGENT_RUN_TIMEOUT_S = 1800

ResponseFormatType: TypeAlias = type[SchemaT]
RT = TypeVar("RT", bound=ResponseFormatType)

Prompt: TypeAlias = str | list["Prompt"] | dict[str, "Prompt"]


@dataclass
class MCPConnection:
    name: str
    connection: Connection


@dataclass
class CachedMCPData:
    tools: list[BaseTool]
    client: MultiServerMCPClient


_MCP_CACHE: dict[str, CachedMCPData] = {}


class Query:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        tool_budget: int = 20,
        reasoning_effort=_UNSET,
    ) -> None:
        self._mcp_connections: list[MCPConnection] = []
        self._tools: list[BaseTool] = []
        self._system_prompt: Optional[str] = None
        self._strict = True

        config = get_config()
        if model is None:
            self._model = config.agent.code.model
        else:
            self._model = model

        if temperature is None:
            self._temperature = config.agent.code.model_temp
        else:
            self._temperature = temperature

        if reasoning_effort is _UNSET:
            self._reasoning_effort = config.agent.code.reasoning_effort
        else:
            self._reasoning_effort = reasoning_effort

        self._tool_budget = tool_budget

    def with_tool(self, *tool: BaseTool) -> Self:
        self._tools.extend(tool)
        return self

    def with_mcp(self, *mcp_connection: MCPConnection) -> Self:
        self._mcp_connections.extend(mcp_connection)
        return self

    def with_system(self, system_prompt: str) -> Self:
        self._system_prompt = system_prompt
        return self

    # TODO: strict is currently unused with the custom langgraph
    # I am not sure if we still need it.
    def non_strict(self) -> Self:
        self._strict = False
        return self

    @overload
    async def run(self, input: Prompt) -> str: ...

    @overload
    async def run(self, input: Prompt, response_schema: RT) -> RT: ...

    async def run(
        self, input: Prompt, response_schema: Optional[RT] = None
    ) -> RT | str:
        input = prompt_to_md(input)
        tools = await self._get_all_tools()

        # Reasoning/Codex models reject non-default sampling params (temperature,
        # top_p, ...) and are steered via reasoning_effort instead. Only send
        # temperature for non-reasoning models; only send reasoning_effort when
        # the model supports it and a level is configured. max_retries absorbs
        # the intermittent 5xx that Codex + tool-calling can return.
        model_kwargs = {
            "model": self._model,
            "use_responses_api": True,
            "max_retries": 4,
            "timeout": _REQUEST_TIMEOUT_S,
        }
        if is_reasoning_model(self._model):
            if self._reasoning_effort:
                model_kwargs["reasoning_effort"] = self._reasoning_effort
        else:
            model_kwargs["temperature"] = self._temperature

        model = ChatOpenAI(**model_kwargs)

        agent = Agent(
            model,
            tools,
            system_prompt=self._system_prompt,
            response_schema=response_schema,
        )

        # LangGraph's recursion_limit counts super-steps (~2 per tool round:
        # llm_call + tools). Its default of 25 is lower than what tool_budget
        # allows, so an eager agent (e.g. Codex at high reasoning effort) hits
        # the hard cap mid-search before the graph's own tool-budget logic can
        # terminate it. Scale the cap to the budget so the budget governs.
        # Wrap the whole agent run in a hard timeout so a stalled OpenAI/MCP
        # await cannot hang the process indefinitely; the caller's retry/scoring
        # logic then handles the resulting TimeoutError.
        try:
            resp = await asyncio.wait_for(
                agent.app.ainvoke(
                    {
                        "messages": [HumanMessage(input)],
                        "tool_budget": self._tool_budget,
                        "structured_response": None,
                    },
                    config={"recursion_limit": 2 * self._tool_budget + 15},
                ),
                timeout=_AGENT_RUN_TIMEOUT_S,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Agent run exceeded {_AGENT_RUN_TIMEOUT_S}s (model={self._model}); "
                "aborting to avoid a hang."
            ) from e

        usage = TokenUsageOpenAi(resp, self._model)
        tracker.add(usage)
        logger.debug(usage)

        if response_schema:
            structured_resp: RT = resp["structured_response"]
            return structured_resp

        messages = resp.get("messages")
        if messages is None or len(messages) == 0:
            raise RuntimeError("LLM did not return any message!")

        # Find the last AIMessage in the conversation
        ai_messages = [msg for msg in reversed(messages) if isinstance(msg, AIMessage)]
        if not ai_messages:
            raise RuntimeError("No AIMessage found in response!")

        return _extract_ai_message_text(ai_messages[0])

    async def _get_all_tools(self) -> list[BaseTool]:
        tools = list(self._tools)
        for mcp in self._mcp_connections:
            if mcp.name in _MCP_CACHE:
                tools.extend(_MCP_CACHE[mcp.name].tools)
            else:
                logger.info(
                    f"Initializing MCP connection and fetching tools for '{mcp.name}'"
                )
                client = MultiServerMCPClient({mcp.name: mcp.connection})
                fetched_tools = await client.get_tools()
                _MCP_CACHE[mcp.name] = CachedMCPData(client=client, tools=fetched_tools)
                tools.extend(fetched_tools)
        return tools


def prompt_to_md(prompt: Prompt) -> str:
    return _prompt_to_md(prompt)[0]


def _prompt_to_md(prompt: Prompt | None, level=1) -> tuple[str, bool]:
    if prompt is None:
        return "None", True
    elif isinstance(prompt, dict):
        parts = []
        any_text = False

        for k, v in prompt.items():
            body, has_text = _prompt_to_md(v, level + 1)
            parts.append(f"{'#' * level} {k}")
            if body:
                parts.append(body)
            if has_text:
                parts.append("")
                any_text = True

        return "\n".join(parts).rstrip(), any_text

    elif isinstance(prompt, list):
        parts = []
        prev_was_text = False
        any_text = False

        for v in prompt:
            body, has_text = _prompt_to_md(v, level)
            if not body:
                continue

            if prev_was_text and body.lstrip().startswith("#"):
                parts.append("")

            parts.append(body)
            prev_was_text = has_text
            any_text |= has_text

        return "\n".join(parts), any_text

    elif isinstance(prompt, str):
        stripped = prompt.strip()
        return stripped, bool(stripped)

    else:
        print(f"Invalid prompt type: {type(prompt)}")
        sys.exit(1)


def _extract_ai_message_text(message: AIMessage) -> str:
    """Extract plain text from a LangChain AIMessage across content formats."""

    text_parts: list[str] = []

    # Prefer LangChain's normalized block view when available.
    content_blocks = getattr(message, "content_blocks", None)
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)

    if text_parts:
        return "\n".join(text_parts)

    content = message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)

    if text_parts:
        return "\n".join(text_parts)

    return str(content)
