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
        print("\n" + "="*70)
        print("🌐 WEB SEARCH TRIGGERED")
        print("="*70)
        print(f"📥 Search Query: {query}")
        print("="*70)
        
        try:
            results = self.tool.invoke({"query": query})
            
            if isinstance(results, list):
                print(f"\n✅ Found {len(results)} web results:\n")
                formatted_results = []
                for i, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        title = result.get('title', 'No title')
                        content = result.get('content', '')
                        url = result.get('url', '')
                        
                        # Print summary of each result
                        print(f"  {i}. {title}")
                        print(f"     URL: {url}")
                        print(f"     Content preview: {content[:100]}...")
                        print()
                        
                        formatted_results.append(
                            f"{i}. {title}\n{content}\nSource: {url}\n"
                        )
                
                final_result = "\n".join(formatted_results)
                print("="*70 + "\n")
                return final_result
            
            print(f"📤 Raw Result: {str(results)[:200]}...")
            print("="*70 + "\n")
            return str(results)
            
        except Exception as e:
            print(f"❌ Error during web search: {e}")
            print("="*70 + "\n")
            return f"Web search unavailable: {e}"
                
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

