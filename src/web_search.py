"""Web search tool for finding information online."""

from typing import List, Optional
import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool

from .config import Config


class WebSearchTool:
    """Tool for searching the web for Gloomhaven rule clarifications."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.
        
        Args:
            api_key: Tavily API key. If None, uses Config.TAVILY_API_KEY
        """
        self.api_key = api_key or Config.TAVILY_API_KEY
        self.tool = None
        
        if self.api_key:
            try:
                self.tool = TavilySearchResults(
                    max_results=Config.MAX_SEARCH_RESULTS,
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize Tavily search: {e}")
                self.tool = None
    
    def search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: Search query
            
        Returns:
            Search results as a string
        """
        if self.tool is None:
            return self._fallback_search(query)
        
        try:
            results = self.tool.invoke({"query": query})
            
            if isinstance(results, list):
                formatted_results = []
                for i, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        title = result.get('title', 'No title')
                        content = result.get('content', '')
                        url = result.get('url', '')
                        formatted_results.append(
                            f"{i}. {title}\n{content}\nSource: {url}\n"
                        )
                return "\n".join(formatted_results)
            
            return str(results)
            
        except Exception as e:
            print(f"Error during web search, using fallback: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> str:
        """
        Fallback search using DuckDuckGo Instant Answer API (no API key required).
        This provides a lightweight, dependency-free backup when Tavily is unavailable.
        """
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                    "no_redirect": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            
            parts: List[str] = []
            abstract = (data or {}).get("AbstractText") or (data or {}).get("Abstract")
            if abstract:
                parts.append(f"1. DuckDuckGo Instant Answer\n{abstract}\n")
            
            related = (data or {}).get("RelatedTopics") or []
            # Flatten nested 'Topics' levels if present
            flat_related = []
            for item in related:
                if isinstance(item, dict) and "Topics" in item:
                    flat_related.extend(item.get("Topics", []))
                else:
                    flat_related.append(item)
            
            for i, item in enumerate(flat_related[: Config.MAX_SEARCH_RESULTS], start=2 if abstract else 1):
                if not isinstance(item, dict):
                    continue
                text = item.get("Text") or item.get("Result") or ""
                url = item.get("FirstURL") or ""
                if text or url:
                    parts.append(f"{i}. {text}\nSource: {url}\n")
            
            if not parts:
                return "No results found via fallback search."
            return "\n".join(parts)
        
        except Exception as e:
            return (
                "Fallback web search failed. "
                "Please set TAVILY_API_KEY in your environment to enable full web search. "
                f"Details: {e}"
            )
    
    def as_langchain_tool(self) -> Tool:
        """
        Convert to a LangChain Tool.
        
        Returns:
            LangChain Tool instance
        """
        return Tool(
            name="web_search",
            description=(
                "Search the web for Gloomhaven rule clarifications. "
                "Use this when the rulebook doesn't contain enough information. "
                "Input should be a search query."
            ),
            func=self.search
        )

