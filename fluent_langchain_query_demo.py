# Experimental fluent langchain query demo; to be removed before PR

import asyncio
from dataclasses import dataclass
from pprint import pprint

from dotenv import load_dotenv
from langchain.tools import tool

from treesearch.llm.query import MCPConnection, Query


@tool(description="Tool call to indicate you are done.")
def finish():
    print("Agent finished!")


async def main():
    load_dotenv()

    greet_mcp = MCPConnection(
        "greet_mcp",
        {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "treesearch/mcp/test_server.py"],
        },
    )

    @dataclass
    class Response:
        message: str  # Final message for the user

    resp = (
        await Query()
        .with_mcp(greet_mcp)
        .with_tool(finish)
        .with_system(
            "Always respond in greek, no matter the input language of user or tools!"
        )
        .run(
            "Greet the user 'Bob'. Generate a greeting using the greet tool. Call the finish tool when you are done!",
            Response,
        )
    )

    pprint(resp)


if __name__ == "__main__":
    asyncio.run(main())
