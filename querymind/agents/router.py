#!/usr/bin/env python3
"""
AgentRouter - Smart routing between FastSearch and DeepResearch

Analyzes query characteristics to automatically select the best agent:
- FastSearchAgent: Simple, clear queries → <1s
- DeepResearchAgent: Complex, semantic queries → ~10s or cached

Routing Heuristics (no LLM overhead):
- Query length: Long queries likely need semantic analysis
- Question words: "how", "why", "what" indicate complex intent
- Logical operators: "and", "or", "not" suggest multi-concept
- Punctuation: Commas, semicolons indicate complex structure

Phase: 3b - Multi-Agent System
"""

import time
from typing import Dict, Any, Tuple
from enum import Enum

from base_agent import BaseAgent, SearchResult
from fast_search_agent import FastSearchAgent
from deep_research_agent import DeepResearchAgent


class AgentType(Enum):
    """Agent selection options"""
    FAST = "fast_search"
    DEEP = "deep_research"


class AgentRouter:
    """
    Smart router for agent selection

    Uses fast heuristics (<50ms) to analyze queries and route to the
    appropriate specialized agent.

    Philosophy: **Use the simplest tool that meets requirements**
    - Default to Fast (optimize for speed)
    - Route to Deep only when complexity requires it
    """

    def __init__(self, model: str = "mistral:7b", shared_collection=None):
        """
        Initialize router with both agents

        Args:
            model: Ollama model for DeepResearchAgent
            shared_collection: Optional pre-initialized ChromaDB collection
                             to avoid client conflicts
        """
        self.fast_agent = FastSearchAgent(shared_collection=shared_collection)
        self.deep_agent = DeepResearchAgent(model=model, shared_collection=shared_collection)

        # Router statistics
        self.stats = {
            "total_routed": 0,
            "fast_selected": 0,
            "deep_selected": 0,
            "total_routing_time": 0.0
        }

    def _analyze_query(self, query: str, verbose: bool = False) -> Tuple[AgentType, str]:
        """
        Analyze query and decide which agent to use

        Heuristics (evaluated in order):
        1. Long queries (>10 words) → Deep
        2. Question words → Deep
        3. Logical operators → Deep
        4. Complex punctuation → Deep
        5. Default → Fast

        Args:
            query: Search query
            verbose: Print routing decision

        Returns:
            (agent_type, reason) tuple
        """
        routing_start = time.time()

        query_lower = query.lower()
        words = query.split()

        # Heuristic 1: Long queries likely need semantic analysis
        # UPDATED 2025-10-01: Lowered from >10 to >6 words for better coverage
        if len(words) > 6:
            reason = f"Long query ({len(words)} words) - needs semantic analysis"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 2: Question words indicate complex intent
        question_words = ['how', 'why', 'what', 'when', 'where', 'which', 'who', 'explain', 'describe', 'compare']
        if any(word in query_lower for word in question_words):
            reason = "Question detected - needs intent understanding"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 3: Logical operators suggest multi-concept query
        logical_operators = [' and ', ' or ', ' not ', ' but ']
        if any(op in query_lower for op in logical_operators):
            reason = "Logical operators - multi-concept query"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 4: Complex punctuation indicates structured query
        if query.count(',') >= 2 or ';' in query or '?' in query:
            reason = "Complex punctuation - structured query"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 5: Phrases suggesting comparison or analysis
        analysis_phrases = ['difference between', 'compare', 'versus', 'vs', 'better than', 'best practices', 'pros and cons']
        if any(phrase in query_lower for phrase in analysis_phrases):
            reason = "Analysis/comparison query detected"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 6: Semantic complexity indicators (NEW 2025-10-01)
        # These terms indicate user needs context understanding, not just keyword matching
        semantic_indicators = [
            'implement', 'setup', 'configure', 'build', 'create',
            'best', 'better', 'improve', 'optimize', 'enhance',
            'fix', 'debug', 'troubleshoot', 'resolve',
            'understand', 'learn', 'explain', 'explore', 'research',
            'guide', 'tutorial', 'walkthrough', 'approach', 'strategy',
            'pattern', 'architecture', 'design', 'structure'
        ]
        if any(indicator in query_lower for indicator in semantic_indicators):
            reason = f"Semantic complexity - needs context understanding"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Heuristic 7: Vague or broad terms (NEW 2025-10-01)
        # These indicate user isn't sure exactly what they want
        vague_terms = ['issue', 'problem', 'thing', 'stuff', 'way', 'method', 'solution']
        if any(term in query_lower for term in vague_terms):
            reason = f"Vague query - needs semantic analysis"
            if verbose:
                print(f"[ROUTER] → DeepResearch: {reason}")
            self._update_routing_stats(AgentType.DEEP, routing_start)
            return (AgentType.DEEP, reason)

        # Default: Fast search for simple lookups
        reason = f"Simple query ({len(words)} words) - keyword matching sufficient"
        if verbose:
            print(f"[ROUTER] → FastSearch: {reason}")
        self._update_routing_stats(AgentType.FAST, routing_start)
        return (AgentType.FAST, reason)

    def _update_routing_stats(self, agent_type: AgentType, routing_start: float):
        """Update routing statistics"""
        self.stats["total_routed"] += 1
        if agent_type == AgentType.FAST:
            self.stats["fast_selected"] += 1
        else:
            self.stats["deep_selected"] += 1
        self.stats["total_routing_time"] += time.time() - routing_start

    def search(self, query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
        """
        Route query to appropriate agent and execute search

        Args:
            query: Natural language search query
            n_results: Number of results to return
            verbose: Print routing decision and search progress

        Returns:
            SearchResult with agent_type indicating which agent was used
        """
        # Analyze and route
        agent_type, reason = self._analyze_query(query, verbose)

        # Execute with selected agent
        if agent_type == AgentType.FAST:
            return self.fast_agent.search(query, n_results, verbose)
        else:
            return self.deep_agent.search(query, n_results, verbose)

    def search_fast(self, query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
        """
        Explicitly use FastSearchAgent (bypass router)

        Args:
            query: Search query
            n_results: Number of results
            verbose: Print progress

        Returns:
            SearchResult from FastSearchAgent
        """
        return self.fast_agent.search(query, n_results, verbose)

    def search_deep(self, query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
        """
        Explicitly use DeepResearchAgent (bypass router)

        Args:
            query: Search query
            n_results: Number of results
            verbose: Print progress

        Returns:
            SearchResult from DeepResearchAgent
        """
        return self.deep_agent.search(query, n_results, verbose)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from router and both agents

        Returns:
            Dict with:
            - Router stats (selection counts, routing time)
            - FastSearchAgent stats
            - DeepResearchAgent stats
        """
        stats = {
            "router": self.stats.copy(),
            "fast_agent": self.fast_agent.get_stats(),
            "deep_agent": self.deep_agent.get_stats()
        }

        # Add routing percentages
        if stats["router"]["total_routed"] > 0:
            total = stats["router"]["total_routed"]
            stats["router"]["fast_percentage"] = (stats["router"]["fast_selected"] / total) * 100
            stats["router"]["deep_percentage"] = (stats["router"]["deep_selected"] / total) * 100
            stats["router"]["avg_routing_time"] = stats["router"]["total_routing_time"] / total
        else:
            stats["router"]["fast_percentage"] = 0.0
            stats["router"]["deep_percentage"] = 0.0
            stats["router"]["avg_routing_time"] = 0.0

        return stats

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_routed": 0,
            "fast_selected": 0,
            "deep_selected": 0,
            "total_routing_time": 0.0
        }
        self.fast_agent.reset_stats()
        self.deep_agent.reset_stats()


def main():
    """Example usage and routing demonstration"""
    import sys

    print("Initializing AgentRouter...")
    router = AgentRouter(model="mistral:7b")

    # Test queries showing different routing decisions
    test_queries = [
        # Should route to FastSearch
        ("Redis caching", "Simple 2-word lookup"),
        ("kubernetes deployment", "Simple term search"),
        ("docker compose", "Simple tool name"),

        # Should route to DeepResearch
        ("How to implement Redis caching for API responses?", "Question with complex intent"),
        ("Explain the difference between StatefulSet and Deployment", "Comparison/analysis"),
        ("What are best practices for microservices deployment?", "Question + multi-concept"),
    ]

    if len(sys.argv) > 1:
        # Use command line query
        query = " ".join(sys.argv[1:])
        test_queries = [(query, "User-provided query")]

    print(f"\nRunning {len(test_queries)} routed searches...\n")

    for query, description in test_queries:
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Description: {description}")
        print(f"{'='*70}")

        result = router.search(query, n_results=3, verbose=True)

        print(f"\nResult:")
        print(f"  Status: {result.status}")
        print(f"  Agent: {result.agent_type}")
        print(f"  Time: {result.elapsed_time:.3f}s")
        print(f"  Results: {result.result_count}")

        if result.error:
            print(f"  Error: {result.error}")

        if result.results:
            print("\n  Top result:")
            r = result.results[0]
            print(f"    File: {r.get('file', 'unknown')}")
            print(f"    Score: {r.get('score', 0):.4f}")
            print(f"    Cached: {r.get('cached', False)}")

        print()

    # Print comprehensive statistics
    print(f"{'='*70}")
    print("ROUTING STATISTICS")
    print(f"{'='*70}")
    stats = router.get_stats()

    print(f"\nRouter Decisions:")
    print(f"  Total queries:      {stats['router']['total_routed']}")
    print(f"  FastSearch:         {stats['router']['fast_selected']} ({stats['router']['fast_percentage']:.1f}%)")
    print(f"  DeepResearch:       {stats['router']['deep_selected']} ({stats['router']['deep_percentage']:.1f}%)")
    print(f"  Avg routing time:   {stats['router']['avg_routing_time']*1000:.1f}ms")

    print(f"\nFastSearch Performance:")
    print(f"  Searches:           {stats['fast_agent']['total_searches']}")
    print(f"  Avg time:           {stats['fast_agent']['avg_time']:.3f}s")
    print(f"  Success rate:       {stats['fast_agent'].get('success_rate', 0)*100:.1f}%")

    print(f"\nDeepResearch Performance:")
    print(f"  Searches:           {stats['deep_agent']['total_searches']}")
    print(f"  Avg time:           {stats['deep_agent']['avg_time']:.3f}s")
    print(f"  Success rate:       {stats['deep_agent']['successful_searches']/max(stats['deep_agent']['total_searches'],1)*100:.1f}%")
    print(f"  Gather cache hits:  {stats['deep_agent'].get('gather_cache_hits', 0)}")
    print(f"  Gather hit rate:    {stats['deep_agent'].get('gather_cache_hit_rate', 0)*100:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())