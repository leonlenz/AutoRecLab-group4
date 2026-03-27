from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from treesearch.llm.graph import Agent
from treesearch.llm.query import MCPConnection, Query


def build_test_connection() -> MCPConnection:
    return MCPConnection(
        name="test_server",
        connection={
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "treesearch.mcp.test_server"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise the custom LangGraph agent with a tiny MCP tool."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Falls back to config.toml / project defaults.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature for the direct graph run.",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=3,
        help="Tool budget passed into the custom graph.",
    )
    parser.add_argument(
        "--name",
        default="Graph Tester",
        help="Name that should be passed to the MCP tool.",
    )
    return parser.parse_args()


def render_message(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        content = json.dumps(content, indent=2, ensure_ascii=True)

    parts = [f"{message.__class__.__name__}: {content!r}"]

    if isinstance(message, AIMessage) and message.tool_calls:
        parts.append(f"tool_calls={json.dumps(message.tool_calls, indent=2)}")

    if isinstance(message, ToolMessage):
        parts.append(f"tool_name={message.name!r}")

    return " | ".join(parts)


async def get_mcp_tools(connection: MCPConnection):
    query = Query()
    query.with_mcp(connection)
    return await query._get_all_tools()


async def run_direct_agent(
    connection: MCPConnection,
    model_name: str | None,
    temperature: float,
    tool_budget: int,
    name: str,
) -> dict[str, Any]:
    tools = await get_mcp_tools(connection)
    model_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "use_responses_api": True,
    }
    if model_name is not None:
        model_kwargs["model"] = model_name

    model = ChatOpenAI(**model_kwargs)
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are testing a custom LangGraph agent. "
            "You must call the available MCP tool exactly once, then answer with the "
            "tool result and a short note that the tool call succeeded."
        ),
    )

    prompt = (
        f"Call the greet tool with the name '{name}'. "
        "After the tool returns, answer in plain text with the tool output."
    )

    return await agent.app.ainvoke(
        {
            "messages": [HumanMessage(prompt)],
            "tool_budget": tool_budget,
            "structured_response": None,
        }
    )


async def run_query_wrapper(
    connection: MCPConnection,
    model_name: str | None,
    tool_budget: int,
    name: str,
) -> str:
    prompt = (
        f"Call the greet tool with the name '{name}'. "
        "Then answer with exactly the string returned by the tool."
    )

    return await (
        Query(model=model_name, temperature=0.0, tool_budget=tool_budget)
        .with_mcp(connection)
        .with_system(
            "You are testing the custom LangGraph wrapper. "
            "Use the tool, then produce a final plain-text answer."
        )
        .run(prompt)
    )


async def main() -> None:
    args = parse_args()
    connection = build_test_connection()

    print("=== Direct Agent Run ===")
    direct_result = await run_direct_agent(
        connection=connection,
        model_name=args.model,
        temperature=args.temperature,
        tool_budget=args.tool_budget,
        name=args.name,
    )

    messages = direct_result.get("messages", [])
    print(f"message_count={len(messages)}")
    print(f"final_tool_budget={direct_result.get('tool_budget')}")
    for idx, message in enumerate(messages, start=1):
        print(f"[{idx}] {render_message(message)}")

    final_ai = next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )
    if final_ai is not None:
        print("\nDirect final AI content:")
        print(final_ai.content)

    print("\n=== Query Wrapper Run ===")
    wrapped_result = await run_query_wrapper(
        connection=connection,
        model_name=args.model,
        tool_budget=args.tool_budget,
        name=args.name,
    )
    print(wrapped_result)


if __name__ == "__main__":
    asyncio.run(main())
