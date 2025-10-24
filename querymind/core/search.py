"""
Core search functionality for QueryMind

This module provides the main search() function that users import.
"""

from typing import List, Dict, Any
from querymind.agents.fast_search_agent import FastSearchAgent
from querymind.agents.base_agent import SearchResult


def search(query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
    """
    Simple semantic search using FastSearch agent

    Args:
        query: Search query string
        n_results: Number of results to return
        verbose: Print detailed progress

    Returns:
        SearchResult with formatted results

    Example:
        >>> from querymind import search
        >>> results = search("Redis caching patterns", n_results=5)
        >>> for r in results.results:
        ...     print(f"{r['file']}: {r['score']}")
    """
    agent = FastSearchAgent()
    return agent.search(query, n_results=n_results, verbose=verbose)
