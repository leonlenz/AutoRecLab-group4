import sys
from dataclasses import dataclass
from typing import Optional, Self, TypeAlias, TypeVar, overload

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, SchemaT
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ProviderStrategy, SchemaT, StructuredOutputValidationError
from .recovery import (StructuredRecoveryPolicy, ensure_structured_agent_response, structured_output)
from langchain.messages import AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphRecursionError
from dataclasses import is_dataclass, fields as dc_fields
from langgraph.errors import GraphRecursionError

from config import get_config
from utils.log import _ROOT_LOGGER
from treesearch.utils.costs_tracker import TokenUsageOpenAi, get_cost_tracker

logger = _ROOT_LOGGER.getChild("llm")
tracker = get_cost_tracker()

ResponseFormatType: TypeAlias = type[SchemaT]
RT = TypeVar("RT", bound=ResponseFormatType)

Prompt: TypeAlias = str | list["Prompt"] | dict[str, "Prompt"]

from collections import deque
from langchain_core.callbacks import BaseCallbackHandler

class ToolOutputBuffer(BaseCallbackHandler):
    """
    Buffers ONLY documentation tool outputs (optional filter) and caps memory
    so we never blow the model context when returning partial results.
    """
    def __init__(
        self,
        tool_name_whitelist: set[str] | None = None,
        max_chunks: int = 12,
        max_total_chars: int = 12_000,
        max_chunk_chars: int = 2_000,
    ) -> None:
        self.tool_name_whitelist = tool_name_whitelist
        self.max_total_chars = max_total_chars
        self.max_chunk_chars = max_chunk_chars
        self._chunks: deque[str] = deque(maxlen=max_chunks)

    @property
    def chunks(self) -> list[str]:
        return list(self._chunks)

    def on_tool_end(self, output, **kwargs) -> None:
        # Try to identify tool name across LC versions
        tool_name = (
            kwargs.get("name")
            or kwargs.get("tool")
            or kwargs.get("tool_name")
            or kwargs.get("serialized", {}).get("name")
        )

        if self.tool_name_whitelist and tool_name not in self.tool_name_whitelist:
            return  # ignore non-doc tools (e.g., ListToolsRequest noise)

        try:
            text = output if isinstance(output, str) else str(output)
        except Exception:
            text = repr(output)

        text = text.strip()
        if not text:
            return

        # cap chunk size
        if len(text) > self.max_chunk_chars:
            text = text[: self.max_chunk_chars] + "\n...[truncated]..."

        self._chunks.append(text)

        # cap total chars
        while sum(len(c) for c in self._chunks) > self.max_total_chars and self._chunks:
            self._chunks.popleft()

def truncate_text(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]..."
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
        max_iterations: int = 25,
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
        self._max_iterations = max_iterations
        
        self._mode = config.local_llm.llm_mode
        self._local_model=config.local_llm.local_model
        self._local_base_url=config.local_llm.base_url
    def with_tool(self, *tool: BaseTool) -> Self:
        self._tools.extend(tool)
        return self

    def with_mcp(self, *mcp_connection: MCPConnection) -> Self:
        self._mcp_connections.extend(mcp_connection)
        return self

    def with_system(self, system_prompt: str) -> Self:
        self._system_prompt = system_prompt
        return self

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

        if response_schema is None:
            response_format = None
        else:
            response_format = ProviderStrategy(response_schema, strict=self._strict)

        # Set up the model based on mode
        if self._mode != "local":
            model = ChatOpenAI(model=self._model, temperature=self._temperature, use_responses_api=True)
        else:
            model = ChatOpenAI(
                model=self._local_model,
                temperature=self._temperature,
                base_url=self._local_base_url,
                api_key="not needed",
                use_responses_api=False, #TODO Try True with new llm models
                
            
            )
        
        agent = create_agent(
            model=model,
            tools=tools,
            response_format=response_format,
            system_prompt=self._system_prompt,
            
            
        )
        # TODO make this more elegant instead of using an large if / else

        # Different execution path for local vs non-local LLMs
        # Recovery policy (tune per your local models)
        if self._mode == "local":
        # policy is only used for local runs
            recovery = StructuredRecoveryPolicy(
                max_attempts=3,
                bad_markers=["commentary to=", "<|", "functions.", "}]**", "ListToolsRequest"],
            )

            tool_buffer = ToolOutputBuffer(
            tool_name_whitelist={"documentation_query"},
            max_chunks=12,
            max_total_chars=12_000,
            max_chunk_chars=2_000,
        )

            try:
                resp = await agent.ainvoke(
                    {"messages": [HumanMessage(input)]},
                    config={"recursion_limit": self._max_iterations},
                )
            except GraphRecursionError:
                logger.warning(
                    "Recursion limit of %d reached. Forcing a direct response without tools.",
                    self._max_iterations,
                )
                forced_input = (
                    input
                    + "\n\n**IMPORTANT**: You have exhausted your allowed tool calls. "
                    "Based on all the research you have already done, provide your "
                    "final answer NOW without calling any more tools."
                )
                fallback_agent = create_agent(
                    model=model,
                    tools=[],
                    response_format=response_format,
                    system_prompt=self._system_prompt,
                )
                resp = await fallback_agent.ainvoke(
                    {"messages": [HumanMessage(forced_input)]},
                    config={"recursion_limit": self._max_iterations},
                )


                usage = TokenUsageOpenAi(resp, self._model)
                tracker.add(usage)
                logger.info(usage)

                if response_schema:
                    try:
                        structured_resp: RT = resp["structured_response"]
                        return structured_resp
                    except (KeyError, StructuredOutputValidationError) as e:
                        logger.warning(f"Structured response failed: {e}. Attempting repair...")

                        repaired = ensure_structured_agent_response(
                            agent_response=resp,
                            schema=response_schema,
                            llm=model,
                            original_prompt=input,
                            policy=recovery,
                        )
                        return repaired

                if not messages:
                    # if we got tool outputs but no final AI message, return partials
                    if tool_buffer.chunks:
                        return "\n\n".join(tool_buffer.chunks)
                    raise RuntimeError("LLM did not return any message!")

                if not ai_messages:
                    if tool_buffer.chunks:
                        return "\n\n".join(tool_buffer.chunks)
                    raise RuntimeError("No AIMessage found in response!")

                return str(ai_messages[0].content)

            except GraphRecursionError:
                logger.warning("Graph recursion limit hit — returning partial tool outputs.")
                return truncate_text("\n\n".join(tool_buffer.chunks), 12_000)

            except Exception as e:
                if response_schema:
                    logger.warning(f"Agent failed: {e}. Trying direct structured output...")
                    try:
                        return structured_output(
                            llm=model,
                            schema=response_schema,
                            task_prompt=input,
                            policy=recovery,
                        )
                    except Exception as repair_error:
                        logger.error(f"Repair also failed: {repair_error}")
                        raise
                raise



    async def _get_all_tools(self) -> list[BaseTool]:
        tools = list(self._tools)
        for mcp in self._mcp_connections:
            if mcp.name in _MCP_CACHE:
                logger.debug(f"CACHE HIT: Using cached tools and connection for MCP '{mcp.name}'")
                tools.extend(_MCP_CACHE[mcp.name].tools)
            else:
                logger.info(f"Initializing MCP connection and fetching tools for '{mcp.name}'")
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
