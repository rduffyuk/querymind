#!/usr/bin/env python3
"""
BaseAgent - Abstract base class for all vault search agents

Defines common interface and shared utilities for specialized agents:
- FastSearchAgent: Quick keyword lookups (<1s)
- DeepResearchAgent: Full semantic analysis (~10s or cached)
- Future agents: CodeSearchAgent, etc.

Phase: 3b - Multi-Agent System
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Global singleton for ChromaDBManager to avoid re-initialization across all agent instances
_CHROMADB_MANAGER_INSTANCE: Optional[Any] = None


@dataclass
class SearchResult:
    """
    Standardized search result format for all agents

    All agents must return this format for consistency across the system.
    """
    query: str                              # Original search query
    results: List[Dict[str, Any]]          # List of search results
    elapsed_time: float                     # Total query time in seconds
    result_count: int                       # Number of results returned
    status: str                             # "success", "no_results", or "error"
    agent_type: str                         # Which agent handled this query
    error: Optional[str] = None            # Error message if status="error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class BaseAgent(ABC):
    """
    Abstract base class for vault search agents

    All agents must implement:
    - search(): Execute query and return SearchResult
    - get_stats(): Return performance metrics

    Shared utilities:
    - _verify_results(): Common result validation logic
    - _format_result(): Standardize result formatting
    """

    def __init__(self, shared_collection=None):
        """
        Initialize base agent with optional shared ChromaDB collection

        Args:
            shared_collection: Optional pre-initialized ChromaDB collection
                             to avoid client conflicts when multiple parts
                             of the system access the same database
        """
        self._shared_collection = shared_collection

    @abstractmethod
    def search(self, query: str, n_results: int = 5, verbose: bool = False) -> SearchResult:
        """
        Execute search query

        Args:
            query: Natural language search query
            n_results: Number of results to return
            verbose: Print detailed progress

        Returns:
            SearchResult with standardized format
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics

        Returns:
            Dict with metrics like:
            - total_searches
            - avg_time
            - cache_hit_rate
            - agent_type
        """
        pass

    def reset_stats(self):
        """Reset performance statistics (optional, implement if needed)"""
        pass

    # Shared utilities

    def _verify_results(self, results: List[Dict[str, Any]], min_score: float = 0.2) -> Dict[str, Any]:
        """
        Common result verification logic

        Checks:
        - Results exist
        - No errors occurred
        - Scores meet minimum threshold

        Args:
            results: Raw search results from ChromaDB
            min_score: Minimum score threshold (0.0-1.0)

        Returns:
            Dict with:
            - valid: bool
            - reason: str
            - results: filtered list
        """
        if not results:
            return {
                "valid": False,
                "reason": "No results found",
                "results": []
            }

        # Check for errors
        if results[0].get("error"):
            return {
                "valid": False,
                "reason": f"Search failed: {results[0]['error']}",
                "results": []
            }

        # Filter low-quality results
        # ChromaDB scores are 1-distance, typically 0.5-0.8 for good matches
        filtered = [r for r in results if r.get("score", 0) > min_score]

        if not filtered:
            print(f"⚠️ Verification filtered all {len(results)} results (min_score={min_score})")
            if results:
                print(f"   Sample scores: {[r.get('score', 0) for r in results[:3]]}")
            return {
                "valid": False,
                "reason": f"No high-confidence results (all scores < {min_score})",
                "results": results  # Return anyway for debugging
            }

        return {
            "valid": True,
            "reason": f"Found {len(filtered)} relevant results",
            "results": filtered
        }

    def _format_result(self, doc: str, meta: Dict, dist: float, cached: bool = False) -> Dict[str, Any]:
        """
        Format a single search result consistently

        Args:
            doc: Document text
            meta: Metadata from ChromaDB
            dist: Distance score from ChromaDB (lower is better)
            cached: Whether this was a cache hit

        Returns:
            Formatted result dict
        """
        return {
            "file": meta.get('file_path', meta.get('file_name', 'unknown')),
            "score": 1 - dist,  # Convert distance to similarity (0-1)
            "content": doc[:200] if len(doc) > 200 else doc,  # Truncated for display
            "full_content": doc,  # Full content for token counting
            "cached": cached
        }

    def _execute_chromadb_search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Shared ChromaDB search implementation with Redis caching

        This is the core search logic used by all agents. It:
        1. Checks Redis cache first (instant if hit)
        2. Falls back to ChromaDB if cache miss
        3. Caches the result for future queries

        Args:
            query_text: Text to search for
            n_results: Number of results

        Returns:
            List of formatted results with cache indicators
        """
        try:
            # Import directly to enable Redis caching
            import sys
            sys.path.insert(0, '/home/rduffy/Documents/Leveling-Life/neural-vault')

            import chromadb
            from chromadb_embeddings_v2 import ChromaDBManager
            from query_cache import get_cache

            # Get cache instance
            cache = get_cache()

            # Try cache first (Phase 2 optimization)
            cached_result = cache.get(query_text, n_results, 'obsidian_vault_mxbai')

            if cached_result:
                # Cache HIT! Convert cached format to agent format
                results = []

                # Handle both cached and fresh ChromaDB response formats
                if 'documents' in cached_result:
                    docs = cached_result['documents'][0] if isinstance(cached_result['documents'], list) else cached_result['documents']
                    metas = cached_result.get('metadatas', [[]])[0] if 'metadatas' in cached_result else []
                    dists = cached_result.get('distances', [[]])[0] if 'distances' in cached_result else []

                    for doc, meta, dist in zip(docs, metas, dists):
                        results.append(self._format_result(doc, meta, dist, cached=True))

                return results

            # Cache MISS - query ChromaDB directly

            # OPTION 1: Use shared collection if provided (avoids client conflict)
            if self._shared_collection is not None:
                collection = self._shared_collection
            else:
                # OPTION 2: Use module-level singleton to avoid reinitializing GPU/embeddings on every call
                global _CHROMADB_MANAGER_INSTANCE

                print(f"DEBUG: Checking singleton - is None? {_CHROMADB_MANAGER_INSTANCE is None}, id={id(_CHROMADB_MANAGER_INSTANCE) if _CHROMADB_MANAGER_INSTANCE else 'None'}")

                if _CHROMADB_MANAGER_INSTANCE is None:
                    try:
                        _CHROMADB_MANAGER_INSTANCE = ChromaDBManager(persist_path='/home/rduffy/Documents/Leveling-Life/neural-vault/chromadb_data')
                    except ValueError as e:
                        # ChromaDB client already exists with different settings (e.g., from production test script)
                        # Fall back to empty results - Redis cache will handle most queries anyway
                        if "already exists" in str(e):
                            return []  # Return empty, let caller handle via Redis cache or error
                        raise  # Re-raise if it's a different ValueError

                collection = _CHROMADB_MANAGER_INSTANCE.client.get_or_create_collection(
                    'obsidian_vault_mxbai',
                    embedding_function=_CHROMADB_MANAGER_INSTANCE.embedding_function
                )

            # Execute query
            chromadb_result = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

            # Cache the result for next time
            cache.set(query_text, n_results, 'obsidian_vault_mxbai', chromadb_result)

            # Convert to agent format
            results = []
            for doc, meta, dist in zip(
                chromadb_result['documents'][0],
                chromadb_result['metadatas'][0],
                chromadb_result['distances'][0]
            ):
                results.append(self._format_result(doc, meta, dist, cached=False))

            return results

        except Exception as e:
            print(f"⚠️ ChromaDB search exception: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return [{"error": str(e), "status": "failed"}]

    def _execute_hybrid_search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search implementation (BM25 + Vector) with feature flag

        Uses existing HybridSearch class from neural-vault.
        Falls back to vector-only if hybrid search fails.

        Args:
            query_text: Text to search for
            n_results: Number of results

        Returns:
            List of formatted results
        """
        try:
            # Check feature flag
            from core.feature_flags import flags
            hybrid_enabled = flags.is_enabled("core.hybrid_search", fallback=False)

            if not hybrid_enabled:
                # Fallback to vector-only search
                return self._execute_chromadb_search(query_text, n_results)

            # Import HybridSearch (lazy import to avoid startup overhead)
            import sys
            sys.path.insert(0, '/home/rduffy/Documents/Leveling-Life/neural-vault')
            from hybrid_search import HybridSearch

            # Initialize hybrid search (will be cached on first use)
            if not hasattr(self, '_hybrid_searcher'):
                self._hybrid_searcher = HybridSearch(
                    chromadb_path='/home/rduffy/Documents/Leveling-Life/neural-vault/chromadb_data',
                    collection_name='obsidian_vault_mxbai',
                    model_name='mixedbread-ai/mxbai-embed-large-v1',
                    device='cuda'
                )

            # Execute hybrid search
            hybrid_results = self._hybrid_searcher.hybrid_search(
                query=query_text,
                top_k=n_results,
                vector_weight=0.5,
                keyword_weight=0.5,
                fusion_method='rrf'
            )

            # Convert SearchResult objects to agent format
            results = []
            for r in hybrid_results:
                results.append({
                    "content": r.content,
                    "file": r.metadata.get('file_path', r.metadata.get('file_name', 'unknown')),
                    "score": r.score,
                    "metadata": r.metadata,
                    "cached": False,
                    "search_type": "hybrid"
                })

            return results

        except Exception as e:
            # Fallback to vector-only on any error
            return self._execute_chromadb_search(query_text, n_results)


class AgentError(Exception):
    """Base exception for agent errors"""
    pass


class GatherError(AgentError):
    """Error during gather phase (intent analysis)"""
    pass


class ActionError(AgentError):
    """Error during action phase (search execution)"""
    pass


class VerifyError(AgentError):
    """Error during verify phase (result validation)"""
    pass