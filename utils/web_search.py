from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["TAVILY_API_KEY"] = "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"
web_search_tool = TavilySearchResults(k=3)
