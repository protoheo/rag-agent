from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional

from langchain_core.tools import tool

try:
    from tavily import TavilyClient
except Exception as e:
    raise RuntimeError(
        "tavily 패키지가 필요합니다. 설치: pip install tavily-python"
    ) from e

# LangGraph에 바로 붙일 수 있는 ToolNode는 선택적으로 제공합니다.
try:
    from langgraph.prebuilt import ToolNode  # type: ignore
except Exception:
    ToolNode = None  # type: ignore


# 내부적으로 Tavily 클라이언트를 생성/캐시
_client: Optional[TavilyClient] = None
_client_key: Optional[str] = None


def _get_client(api_key: Optional[str] = None) -> TavilyClient:
    """
    TAVILY_API_KEY 환경변수 또는 전달된 api_key로 TavilyClient를 반환합니다.
    """
    global _client, _client_key
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("환경변수 TAVILY_API_KEY가 설정되지 않았습니다.")
    if _client is None or _client_key != key:
        _client = TavilyClient(api_key=key)
        _client_key = key
    return _client


@tool("tavily_search")
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = False,
) -> Dict[str, object]:
    """
    Tavily 웹 검색 Tool.
    - query: 검색 질의문
    - max_results: 결과 개수
    - search_depth: 'basic' | 'advanced'
    - include_answer: Tavily가 생성한 요약(answer) 포함 여부

    반환 형식:
    {
        "results": [
            {"title": str, "url": str, "snippet": str},
            ...
        ],
        "answer": Optional[str]
    }
    """
    client = _get_client()
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
    )

    items: List[Dict[str, str]] = []
    for r in resp.get("results", []):
        items.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or "",
            }
        )

    return {
        "results": items,
        "answer": resp.get("answer"),
    }


def temp_tool():
    with open("temp.txt", "r") as f:
        file = f.read()


def get_tools():
    """
    LangGraph/LLM에 등록할 Tool 리스트를 반환합니다.
    """
    return [tavily_search]


def get_tool_node(tools=None):
    """
    LangGraph에서 사용할 ToolNode를 반환합니다.
    ToolNode가 import 불가한 환경이면 예외를 발생시킵니다.
    """
    if ToolNode is None:
        raise RuntimeError(
            "langgraph가 설치되어 있지 않습니다. 설치: pip install langgraph"
        )
    return ToolNode(tools or [tavily_search])


__all__ = ["tavily_search", "get_tools", "get_tool_node"]
