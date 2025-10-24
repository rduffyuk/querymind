#!/usr/bin/env python3
"""
DeepResearchAgent - Full semantic analysis with Ollama LLM + Web Fallback

Uses complete Gather→Action→Verify workflow with Ollama intent analysis.
Optimized with dual-layer caching (gather cache + action cache).

Performance:
- First query: ~10-14s (10s gather + 4s action)
- Cached query: <0.1s (both caches hit)
- Similar query: ~4s (gather cache hit, action cache miss)
- Web fallback: +2-5s when vault has no results

Best for:
- Complex questions: "How to implement Redis caching?"
- Vague queries: "speed up database queries"
- Multi-concept: "deployment patterns for microservices"

Phase: 3b+ - Multi-Agent System with Web Search
"""

import time
from typing import Dict, List, Any, Optional

from base_agent import BaseAgent, SearchResult
from vault_search_agent_local import VaultSearchAgentLocal
from web_search_client import WebSearchClient


class DeepResearchAgent(BaseAgent):
    """
    Deep research agent - full Ollama semantic analysis with web fallback

    Strategy:
    1. Gather: Ollama LLM intent analysis (with cache)
    2. Action: ChromaDB semantic search (with cache)
    3. Verify: Quality filtering and validation
    4. Fallback: Web search if no vault results (optional)

    This is a thin wrapper around VaultSearchAgentLocal with added
    web search fallback capability.
    """

    def __init__(self, model: str = "mistral:7b", enable_web_fallback: bool = True, shared_collection=None):
        """
        Initialize deep research agent

        Args:
            model: Ollama model for intent analysis
            enable_web_fallback: Enable web search when vault returns no results
            shared_collection: Optional pre-initialized ChromaDB collection
        """
        # Call parent class to set up shared collection
        super().__init__(shared_collection)

        # Delegate to existing optimized agent
        self.agent = VaultSearchAgentLocal(model=model)

        # Web search fallback
        self.enable_web_fallback = enable_web_fallback
        self.web_client = WebSearchClient() if enable_web_fallback else None

        # Track web fallback usage
        self.web_fallback_count = 0

    def search(self, query: str, n_results: int = 5, verbose: bool = False, use_web_fallback: Optional[bool] = None) -> SearchResult:
        """
        Execute deep research search (full Gather→Action→Verify with web fallback)

        Flow:
        1. Gather: Ollama intent analysis (cached if repeat query)
        2. Action: ChromaDB search (cached if repeat search)
        3. Verify: Result validation
        4. Fallback: If no results, try web search (optional)

        Args:
            query: Natural language search query
            n_results: Number of results to return
            verbose: Print detailed progress
            use_web_fallback: Override default web fallback setting for this query

        Returns:
            SearchResult with agent_type="deep_research" or "deep_research_web"
        """
        # Determine if web fallback should be used for this query
        web_fallback_enabled = use_web_fallback if use_web_fallback is not None else self.enable_web_fallback

        # Delegate to VaultSearchAgentLocal
        result = self.agent.search(query, n_results, verbose)

        # Override agent_type to indicate this came from DeepResearchAgent
        result.agent_type = "deep_research"

        # Check if we should try web fallback
        if web_fallback_enabled and result.status == "no_results" and self.web_client:
            if verbose:
                print(f"[WEB FALLBACK] No vault results found, searching web...")

            try:
                # Search the web
                web_results = self.web_client.search_sync(query, n_results)

                # Convert web results to vault-compatible format
                formatted_results = []
                for web_result in web_results:
                    formatted_results.append({
                        "file": web_result.url,
                        "score": 1.0 - (web_result.position * 0.1),  # Score decreases by position
                        "content": f"{web_result.title}\n\n{web_result.snippet}",
                        "cached": False,
                        "source": "web",
                        "web_source": web_result.source
                    })

                if formatted_results:
                    # Update result with web results
                    result.results = formatted_results
                    result.result_count = len(formatted_results)
                    result.status = "success"
                    result.agent_type = "deep_research_web"
                    result.error = None

                    self.web_fallback_count += 1

                    if verbose:
                        print(f"[WEB FALLBACK] Found {len(formatted_results)} web results")

            except Exception as e:
                if verbose:
                    print(f"[WEB FALLBACK] Error: {e}")
                # Keep original no_results status
                pass

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics

        Returns:
            Stats dict with:
            - Total searches
            - Phase timings (gather, action, verify)
            - Cache hit rates (gather cache, action cache)
            - Web fallback usage
            - Success rate
        """
        stats = self.agent.get_stats()
        stats["agent_type"] = "deep_research"
        stats["web_fallback_count"] = self.web_fallback_count
        stats["web_fallback_enabled"] = self.enable_web_fallback

        # Add web search stats if available
        if self.web_client:
            stats["web_search_stats"] = self.web_client.get_stats()

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.agent.reset_stats()


def main():
    """Example usage and basic testing"""
    import sys

    print("Initializing DeepResearchAgent (using Ollama)...")
    agent = DeepResearchAgent(model="mistral:7b")

    # Test queries (complex/semantic)
    test_queries = [
        "How to implement Redis caching for API responses?",
        "Explain the difference between StatefulSet and Deployment",
        "What are best practices for microservices deployment?"
    ]

    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    print(f"\nRunning {len(test_queries)} deep research searches...\n")

    for query in test_queries:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = agent.search(query, n_results=3, verbose=True)

        print(f"\nStatus: {result.status}")
        print(f"Agent: {result.agent_type}")
        print(f"Time: {result.elapsed_time:.3f}s")
        print(f"Results: {result.result_count}")

        if result.error:
            print(f"Error: {result.error}")

        if result.results:
            print("\nTop results:")
            for i, r in enumerate(result.results[:3], 1):
                print(f"\n{i}. {r.get('file', 'unknown')}")
                print(f"   Score: {r.get('score', 0):.4f}")
                print(f"   Cached: {r.get('cached', False)}")
                print(f"   Content: {r.get('content', '')[:100]}...")

        print()

    # Print statistics with phase breakdown
    print(f"{'='*60}")
    print("DEEP RESEARCH STATISTICS")
    print(f"{'='*60}")
    stats = agent.get_stats()
    print(f"\nOverall:")
    print(f"  Total searches:     {stats['total_searches']}")
    print(f"  Successful:         {stats['successful_searches']}")
    print(f"  Failed:             {stats['failed_searches']}")
    print(f"  Success rate:       {stats['successful_searches']/max(stats['total_searches'],1)*100:.1f}%")
    print(f"\nTiming (averages):")
    print(f"  Total time:         {stats['avg_time']:.3f}s")
    print(f"  Gather phase:       {stats.get('avg_gather_time', 0):.3f}s")
    print(f"  Action phase:       {stats.get('avg_action_time', 0):.3f}s")
    print(f"  Verify phase:       {stats.get('avg_verify_time', 0):.3f}s")
    print(f"\nCache Performance:")
    print(f"  Gather cache hits:  {stats.get('gather_cache_hits', 0)}")
    print(f"  Gather cache miss:  {stats.get('gather_cache_misses', 0)}")
    print(f"  Gather hit rate:    {stats.get('gather_cache_hit_rate', 0)*100:.1f}%")
    print(f"  Action hit rate:    {stats.get('action_cache_hit_rate', 0)*100:.1f}%")
    print(f"\nAgent Type: {stats['agent_type']}")

    return 0


if __name__ == "__main__":
    exit(main())