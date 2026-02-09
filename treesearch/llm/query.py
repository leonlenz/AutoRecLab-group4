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

from dataclasses import is_dataclass, fields as dc_fields

from config import get_config
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("llm")

ResponseFormatType: TypeAlias = type[SchemaT]
RT = TypeVar("RT", bound=ResponseFormatType)

Prompt: TypeAlias = str | list["Prompt"] | dict[str, "Prompt"]


@dataclass
class MCPConnection:
    name: str
    connection: Connection


""" # TODO:
- [x] limit number of tool calls (solved: implemented via max_iterations parameter)
- [ ] the error handling in lines 75 - 80 is prob not ideal 
"""


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
        #TODO REMOVE HARD CODE
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
            response_format = ProviderStrategy(response_schema, strict=False)

        # Set up the model based on mode
        if self._mode != "local":
            model = ChatOpenAI(model=self._model, temperature=self._temperature)
        else:
            model = ChatOpenAI(
                model=self._local_model,
                temperature=self._temperature,
                base_url=self._local_base_url,
                api_key="not needed",
                use_responses_api=False,
                
            
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

            try:
                resp = await agent.ainvoke(
                    {"messages": [HumanMessage(input)]},
                    config={"recursion_limit": self._max_iterations},
                )

                messages = resp.get("messages") or []
                ai_messages = [m for m in reversed(messages) if isinstance(m, AIMessage)]
                if ai_messages:
                    last = ai_messages[0]
                    print("\n==== RAW AIMessage.content (type) ====")
                    print(type(last.content))
                    print("==== RAW AIMessage.content (value) ====")
                    print(repr(last.content))
                    print("=====================================\n")
                else:
                    print("\n==== NO AIMessage RETURNED ====\n")

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
                    raise RuntimeError("LLM did not return any message!")

                if not ai_messages:
                    raise RuntimeError("No AIMessage found in response!")

                return str(ai_messages[0].content)

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
        tools = self._tools

        connection_dict: dict[str, Connection] = {
            mcp.name: mcp.connection for mcp in self._mcp_connections
        }
        client = MultiServerMCPClient(connection_dict)
        tools.extend(await client.get_tools())

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
