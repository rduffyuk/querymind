#!/usr/bin/env python3
"""
WebSearchClient - Basic Serper.dev API integration for web search

Provides web search fallback when vault has no results.
Simplified version without rate limiting or retry logic.

API Setup:
1. Sign up at https://serper.dev
2. Get API key from dashboard
3. Set environment variable: export SERPER_API_KEY='your-key'

Cost:
- First 100 queries/month: FREE
- After free tier: $0.30 per 1,000 queries
"""

import os
import time
from typing import List, Optional
from dataclasses import dataclass
from querymind.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WebSearchResult:
    """Single web search result from Serper.dev"""
    title: str
    url: str
    snippet: str
    position: int
    source: str = "serper"


class WebSearchClient:
    """
    Simple web search client using Serper.dev API.

    No advanced features in this simplified version:
    - No rate limiting
    - No retry logic
    - No connection pooling
    - Synchronous only

    Example:
        >>> client = WebSearchClient(api_key="your-key")
        >>> results = client.search_sync("Kubernetes deployment patterns", n_results=5)
        >>> for r in results:
        ...     print(f"{r.title}: {r.url}")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search client.

        Args:
            api_key: Serper.dev API key. If None, reads from SERPER_API_KEY env var
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")

        if not self.api_key:
            logger.warning("No Serper API key found. Set SERPER_API_KEY environment variable.")
            logger.warning("Web search will fail until configured.")

        self.api_url = "https://google.serper.dev/search"
        logger.info("WebSearchClient initialized")

    def search_sync(
        self,
        query: str,
        n_results: int = 5
    ) -> List[WebSearchResult]:
        """
        Search the web synchronously (blocking).

        Args:
            query: Search query
            n_results: Number of results to return (max 10)

        Returns:
            List of WebSearchResult objects

        Raises:
            Exception: If API key missing or request fails
        """
        if not self.api_key:
            raise ValueError("Serper API key not configured. Set SERPER_API_KEY environment variable.")

        # Limit to Serper.dev max
        n_results = min(n_results, 10)

        start_time = time.time()
        logger.info(f"Web search: '{query[:50]}...' (n={n_results})")

        try:
            import requests

            # Call Serper.dev API
            response = requests.post(
                self.api_url,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": n_results
                },
                timeout=10  # 10 second timeout
            )

            if response.status_code == 200:
                data = response.json()
                results = self._parse_results(data)

                elapsed = time.time() - start_time
                logger.info(f"Web search complete: {len(results)} results in {elapsed:.2f}s")

                return results

            elif response.status_code == 429:
                logger.error("Serper API rate limit exceeded")
                raise Exception("Rate limit exceeded. Check your Serper.dev quota.")

            elif response.status_code == 401:
                logger.error("Serper API authentication failed")
                raise Exception("Invalid API key. Check SERPER_API_KEY.")

            else:
                logger.error(f"Serper API error: status={response.status_code}")
                raise Exception(f"Web search failed: HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error("Web search timeout after 10s")
            raise Exception("Web search timeout")

        except Exception as e:
            logger.error(f"Web search error: {e}")
            raise

    def _parse_results(self, data: dict) -> List[WebSearchResult]:
        """
        Parse Serper.dev API response into WebSearchResult objects.

        Args:
            data: JSON response from Serper.dev API

        Returns:
            List of parsed WebSearchResult objects
        """
        results = []

        # Extract organic search results
        organic = data.get("organic", [])

        for idx, item in enumerate(organic):
            result = WebSearchResult(
                title=item.get("title", "No title"),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=idx + 1,
                source="serper"
            )
            results.append(result)

        return results

    def health_check(self) -> bool:
        """
        Check if Serper.dev API is accessible.

        Returns:
            True if API is working, False otherwise
        """
        if not self.api_key:
            logger.warning("Health check failed: No API key")
            return False

        try:
            # Simple test query
            results = self.search_sync("test", n_results=1)
            logger.info("Health check passed")
            return len(results) > 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Convenience function for quick testing
def quick_search(query: str, n_results: int = 5) -> List[WebSearchResult]:
    """
    Quick search function for testing.

    Args:
        query: Search query
        n_results: Number of results

    Returns:
        List of search results

    Example:
        >>> from querymind.agents.web_search_client import quick_search
        >>> results = quick_search("Python FastAPI tutorial")
        >>> print(results[0].title)
    """
    client = WebSearchClient()
    return client.search_sync(query, n_results)
