import os
from datetime import datetime, timezone
from typing import List, Dict

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def web_search(
    query: str,
    max_results: int = 5,
) -> List[Dict]:
    """
    Web Research Tool.

    Uses Tavily to retrieve web sources and converts
    provider-specific results into our standard
    VMC AI OS evidence format.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured in .env"
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
    )

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    results = []

    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score"),
                "retrieved_at": retrieved_at,
            }
        )

    return results
