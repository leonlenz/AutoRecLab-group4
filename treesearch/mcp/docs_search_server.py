from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from config import get_config

load_dotenv()
mcp = FastMCP("Documentation search")

config = get_config()

if config.local_llm.embedding_mode == "api":
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

else:
    embedding_model = OpenAIEmbeddings(
        model=config.local_llm.local_embedding_model,
        base_url=config.local_llm.base_url,
        api_key="not needed",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
            )

VECTOR_STORES_BASE_PTH = Path("./ragEmbeddings")
VECTOR_STORE_NAMES = Literal["omnirec", "lenskit", "recbole"]


def load_vector_store(name: str) -> FAISS:
    vector_store_pth = VECTOR_STORES_BASE_PTH / name
    if not vector_store_pth.exists():
        raise FileNotFoundError(
            f"Could not read in store at '{vector_store_pth}'! Did you generate or download the embeddings first?"
        )
    return FAISS.load_local(
        str(vector_store_pth),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


VECTOR_STORES: dict[str, FAISS] = {
    name: load_vector_store(name) for name in get_args(VECTOR_STORE_NAMES)
}


@mcp.tool()
def documentation_query(library: VECTOR_STORE_NAMES, query: str, k: int = 4) -> str:
    """Queries the documentation and code of a given library

    Args:
        library (str): Target documentation store to search. Needs to be one of: ["omnirec", "lenskit", "recbole"]
        query (str): Natural-language search query.
        k (int, optional): Number of top matching documents to return. Defaults to 4.

    Returns:
        str: Top-k relevant documentation entries formatted in a string.
    """
    vector_store = VECTOR_STORES.get(library)
    if vector_store is None:
        return f"Invalid library! The library parameter needs to be one of: {VECTOR_STORE_NAMES}"

    results = vector_store.similarity_search_with_score(query, k)

    final_output = f"Found {len(results)} relevant documentation sections:\n\n"
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        # Convert distance to similarity score (lower distance = higher similarity)
        similarity = 1 / (1 + score)  # Simple conversion

        final_output += f"--- Result {i} (Relevance: {similarity:.2%}) ---\n"
        final_output += f"Source: {source}\n"
        final_output += f"Content:\n{doc.page_content}\n\n"

    return final_output


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
