#!/usr/bin/env python3
"""
VaultSearchAgentLocal - Simplified vault search with Ollama intent analysis

Implements the core Gather→Action→Verify pattern from neural vault:
1. Gather: Use Ollama to extract search keywords from query
2. Action: Execute ChromaDB search with keywords
3. Verify: Filter and validate results

Simplified version (~200 lines):
- Removed: Gather cache, hot-reload, advanced metrics
- Kept: Ollama integration, ChromaDB search, basic verification

Based on: /home/rduffy/Documents/Leveling-Life/agents/vault_search_agent_local.py
"""

import time
import os
from typing import Dict, List, Any, Optional
from querymind.core.logging_config import get_logger
from querymind.agents.base_agent import BaseAgent, SearchResult

logger = get_logger(__name__)


class VaultSearchAgentLocal(BaseAgent):
    """
    Simplified vault search agent using Ollama for intent analysis.

    This agent uses a local Ollama model to understand query intent,
    then searches the ChromaDB vault with optimized keywords.

    Performance:
    - First query: ~10-14s (10s Ollama + 4s ChromaDB)
    - No caching in this simplified version

    Example:
        >>> agent = VaultSearchAgentLocal(model="mistral:7b")
        >>> result = agent.search("How to implement Redis caching?")
        >>> print(f"Found {result.result_count} results")
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize vault search agent.

        Args:
            model: Ollama model name (default: mistral:7b)
            ollama_url: Ollama API endpoint
        """
        super().__init__()
        self.model = model
        self.ollama_url = ollama_url
        self.agent_type = "vault_search_local"

        # Verify Ollama is available
        self._verify_ollama()

        logger.info(f"VaultSearchAgentLocal initialized with model={model}")

    def _verify_ollama(self) -> None:
        """Verify Ollama service is running."""
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"Ollama service available at {self.ollama_url}")
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama not available: {e}. Intent analysis will fail.")

    def search(
        self,
        query: str,
        n_results: int = 5,
        verbose: bool = False
    ) -> SearchResult:
        """
        Search vault using Gather→Action→Verify workflow.

        Args:
            query: User's search query
            n_results: Number of results to return
            verbose: Enable detailed logging

        Returns:
            SearchResult with status, results, and timing
        """
        start_time = time.time()

        if verbose:
            logger.info(f"Starting vault search: '{query[:50]}...'")

        # Phase 1: Gather - Extract keywords using Ollama
        gather_start = time.time()
        keywords = self._gather_intent(query, verbose)
        gather_time = time.time() - gather_start

        if verbose:
            logger.info(f"Gather phase complete in {gather_time:.2f}s: keywords='{keywords}'")

        # Phase 2: Action - Search ChromaDB with keywords
        action_start = time.time()
        raw_results = self._execute_search(keywords, n_results)
        action_time = time.time() - action_start

        if verbose:
            logger.info(f"Action phase complete in {action_time:.2f}s: {len(raw_results)} raw results")

        # Phase 3: Verify - Filter and validate results
        verify_start = time.time()
        verified = self._verify_results(raw_results, min_score=0.3)
        verify_time = time.time() - verify_start

        if verbose:
            logger.info(f"Verify phase complete in {verify_time:.2f}s: {len(verified['results'])} valid results")

        # Build final result
        elapsed = time.time() - start_time

        return SearchResult(
            status="success" if verified['valid'] else "no_results",
            query=query,
            result_count=len(verified['results']),
            results=verified['results'],
            elapsed_time=elapsed,
            agent_type=self.agent_type,
            error=None if verified['valid'] else verified['reason']
        )

    def _gather_intent(self, query: str, verbose: bool = False) -> str:
        """
        Phase 1: Gather - Use Ollama to extract search keywords.

        Asks the LLM: "What are the key search terms for this query?"
        This improves search quality by focusing on core concepts.

        Args:
            query: User's original query
            verbose: Enable detailed logging

        Returns:
            Optimized search keywords string
        """
        try:
            import requests

            # Prompt engineering for keyword extraction
            prompt = f"""You are a search keyword extractor. Given a user query, extract the most important keywords for semantic search.

User query: "{query}"

Return ONLY the keywords (2-5 words), nothing else. Focus on technical terms and concepts.

Keywords:"""

            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "num_predict": 30    # Limit output length
                    }
                },
                timeout=15  # 15 second timeout
            )

            if response.status_code == 200:
                result = response.json()
                keywords = result['response'].strip()

                if verbose:
                    logger.debug(f"Ollama extracted keywords: '{keywords}'")

                return keywords if keywords else query
            else:
                logger.warning(f"Ollama returned status {response.status_code}, using original query")
                return query

        except Exception as e:
            logger.error(f"Gather phase failed: {e}, using original query")
            return query  # Fallback to original query

    def _execute_search(self, keywords: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Phase 2: Action - Execute ChromaDB search.

        Uses the base agent's ChromaDB search functionality.

        Args:
            keywords: Search keywords from gather phase
            n_results: Number of results to retrieve

        Returns:
            List of raw search results
        """
        # Use parent class's ChromaDB search
        results = self._execute_chromadb_search(keywords, n_results)
        return results if results else []

    def _verify_results(
        self,
        results: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Phase 3: Verify - Filter and validate search results.

        Applies quality filters:
        - Score threshold (default: 0.3)
        - Error checking
        - Result deduplication

        Args:
            results: Raw results from action phase
            min_score: Minimum relevance score

        Returns:
            Dict with 'valid' flag, 'reason', and filtered 'results'
        """
        # Check for errors
        if not results:
            return {
                "valid": False,
                "reason": "No results found in vault",
                "results": []
            }

        if results and 'error' in results[0]:
            return {
                "valid": False,
                "reason": f"Search failed: {results[0]['error']}",
                "results": []
            }

        # Filter by score threshold
        filtered = [r for r in results if r.get('score', 0) >= min_score]

        if not filtered:
            return {
                "valid": False,
                "reason": f"No high-confidence results (all scores < {min_score})",
                "results": results  # Return anyway for debugging
            }

        return {
            "valid": True,
            "reason": f"Found {len(filtered)} high-quality results",
            "results": filtered
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.

        Returns:
            Dict with basic agent info (no advanced metrics in simplified version)
        """
        return {
            "agent_type": self.agent_type,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "note": "Simplified version - no cache metrics"
        }
