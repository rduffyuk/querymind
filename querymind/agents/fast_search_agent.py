#!/usr/bin/env python3
"""
FastSearchAgent - High-speed search without LLM overhead

Optimized for simple, clear queries where keyword matching is sufficient.
Skips Ollama gather phase entirely, using fast regex-based keyword extraction.

Performance:
- Target: <1s response time
- No gather phase (saves 10s)
- Action cache still active (instant on repeats)

Best for:
- Simple lookups: "Redis caching", "docker compose"
- Known terms: "kubernetes StatefulSet"
- Quick reference: "vim config"

Phase: 3b - Multi-Agent System
"""

import time
import re
from typing import Dict, List, Any, Optional

from base_agent import BaseAgent, SearchResult


class FastSearchAgent(BaseAgent):
    """
    Fast search agent - no LLM, pure keyword matching

    Strategy:
    1. Skip gather phase (no Ollama)
    2. Fast keyword extraction (regex)
    3. Direct ChromaDB query with action cache
    4. Minimal verification
    """

    def __init__(self, shared_collection=None):
        """
        Initialize fast search agent

        Args:
            shared_collection: Optional pre-initialized ChromaDB collection
        """
        # Call parent class to set up shared collection
        super().__init__(shared_collection)

        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "extract_time": 0.0,  # Keyword extraction time
            "action_time": 0.0,
            "verify_time": 0.0
        }

        # Stop words to filter out (common words with low semantic value)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

    def _extract_keywords_simple(self, query: str) -> List[str]:
        """
        Fast keyword extraction without LLM

        Strategy:
        1. Lowercase and tokenize
        2. Remove stop words
        3. Filter short words (<3 chars)
        4. Limit to top 5 keywords

        Args:
            query: Search query

        Returns:
            List of extracted keywords (max 5)
        """
        extract_start = time.time()

        # Lowercase and remove punctuation
        normalized = query.lower()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Tokenize
        words = normalized.split()

        # Filter: remove stop words and short words
        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) >= 3
        ]

        # Limit to 5 most important (first 5 words after filtering)
        keywords = keywords[:5]

        extract_elapsed = time.time() - extract_start
        self.stats["extract_time"] += extract_elapsed

        return keywords

    def search(self, query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
        """
        Execute fast search (no gather phase)

        Flow:
        1. Extract keywords (fast regex, no LLM)
        2. Query ChromaDB (with action cache)
        3. Verify results (minimal filtering)

        Args:
            query: Natural language search query
            n_results: Number of results to return
            verbose: Print detailed progress

        Returns:
            SearchResult with agent_type="fast_search"
        """
        start_time = time.time()
        self.stats["total_searches"] += 1

        try:
            # Step 1: Extract keywords (replaces gather phase)
            if verbose:
                print(f"[EXTRACT] Fast keyword extraction: {query}")

            keywords = self._extract_keywords_simple(query)

            if verbose:
                print(f"[EXTRACT] Keywords: {keywords}")

            # Build search query from keywords
            search_query = ' '.join(keywords) if keywords else query

            # Step 2: Action - Execute search (hybrid if enabled, vector-only fallback)
            if verbose:
                print(f"[ACTION] Searching vault (hybrid if enabled)...")

            action_start = time.time()
            results = self._execute_hybrid_search(search_query, n_results)
            action_elapsed = time.time() - action_start
            self.stats["action_time"] += action_elapsed

            if verbose and results and "search_type" in results[0]:
                search_type = results[0].get("search_type", "vector")
                print(f"[ACTION] Used {search_type} search")

            # Step 3: Verify results (minimal)
            if verbose:
                print(f"[VERIFY] Validating {len(results)} results...")

            verify_start = time.time()
            verification = self._verify_results(results, min_score=0.2)
            verify_elapsed = time.time() - verify_start
            self.stats["verify_time"] += verify_elapsed

            elapsed = time.time() - start_time

            # Update stats
            if verification["valid"]:
                self.stats["successful_searches"] += 1
            else:
                self.stats["failed_searches"] += 1

            self.stats["total_time"] += elapsed
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_searches"]

            return SearchResult(
                query=query,
                results=verification["results"],
                elapsed_time=elapsed,
                result_count=len(verification["results"]),
                status="success" if verification["valid"] else "no_results",
                agent_type="fast_search",
                error=None if verification["valid"] else verification["reason"]
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.stats["failed_searches"] += 1
            self.stats["total_time"] += elapsed
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_searches"]

            return SearchResult(
                query=query,
                results=[],
                elapsed_time=elapsed,
                result_count=0,
                status="error",
                agent_type="fast_search",
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        stats["agent_type"] = "fast_search"

        if stats["total_searches"] > 0:
            stats["avg_extract_time"] = stats["extract_time"] / stats["total_searches"]
            stats["avg_action_time"] = stats["action_time"] / stats["total_searches"]
            stats["avg_verify_time"] = stats["verify_time"] / stats["total_searches"]
            stats["success_rate"] = stats["successful_searches"] / stats["total_searches"]
        else:
            stats["avg_extract_time"] = 0.0
            stats["avg_action_time"] = 0.0
            stats["avg_verify_time"] = 0.0
            stats["success_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "extract_time": 0.0,
            "action_time": 0.0,
            "verify_time": 0.0
        }


def main():
    """Example usage and basic testing"""
    import sys

    print("Initializing FastSearchAgent...")
    agent = FastSearchAgent()

    # Test queries (simple lookups)
    test_queries = [
        "Redis caching",
        "kubernetes deployment",
        "docker compose"
    ]

    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    print(f"\nRunning {len(test_queries)} fast searches...\n")

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

    # Print statistics
    print(f"{'='*60}")
    print("FAST SEARCH STATISTICS")
    print(f"{'='*60}")
    stats = agent.get_stats()
    print(f"\nOverall:")
    print(f"  Total searches:     {stats['total_searches']}")
    print(f"  Successful:         {stats['successful_searches']}")
    print(f"  Failed:             {stats['failed_searches']}")
    print(f"  Success rate:       {stats.get('success_rate', 0)*100:.1f}%")
    print(f"\nTiming (averages):")
    print(f"  Total time:         {stats['avg_time']:.3f}s")
    print(f"  Extract phase:      {stats.get('avg_extract_time', 0):.3f}s")
    print(f"  Action phase:       {stats.get('avg_action_time', 0):.3f}s")
    print(f"  Verify phase:       {stats.get('avg_verify_time', 0):.3f}s")
    print(f"\nAgent Type: {stats['agent_type']}")

    return 0


if __name__ == "__main__":
    exit(main())