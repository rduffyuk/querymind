#!/usr/bin/env python3
"""
Smoke Test: Router Logic

Verifies that the AgentRouter correctly routes queries to appropriate agents
based on query characteristics (length, complexity, question words).

Run with: pytest tests/test_router_basic.py -v
"""

import pytest
from querymind.agents.router import AgentRouter, AgentType


@pytest.fixture
def router():
    """Create a router instance for testing."""
    return AgentRouter()


def test_router_initialization(router):
    """Test that router initializes correctly."""
    assert router is not None
    assert hasattr(router, 'fast_agent')
    assert hasattr(router, 'deep_agent')


def test_simple_query_routes_to_fast(router):
    """Test that simple queries route to FastSearchAgent."""
    simple_queries = [
        "Redis caching",
        "kubernetes StatefulSet",
        "vim config",
        "journal automation"
    ]

    for query in simple_queries:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type == AgentType.FAST, f"Query '{query}' should route to FAST agent, got {agent_type}"
        assert "simple" in reason.lower() or "fast" in reason.lower()


def test_complex_query_routes_to_deep(router):
    """Test that complex queries route to DeepResearchAgent."""
    complex_queries = [
        "How to implement Redis caching for APIs?",
        "Explain StatefulSet vs Deployment differences",
        "What are best practices for error handling?",
        "Describe the journal automation architecture",
        "Why does my function return undefined instead of the expected value?"
    ]

    for query in complex_queries:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type == AgentType.DEEP, f"Query '{query}' should route to DEEP agent, got {agent_type}"
        # Reason should indicate why it went to DEEP (complex, deep, question, or long)
        assert any(keyword in reason.lower() for keyword in ["complex", "deep", "question", "long"])


def test_long_query_routes_to_deep(router):
    """Test that long queries (>10 words) route to DeepResearchAgent."""
    long_query = "I need to understand how the system handles caching when multiple users are accessing the same data simultaneously"

    agent_type, reason = router._analyze_query(long_query, verbose=False)
    assert agent_type == AgentType.DEEP
    assert "long" in reason.lower() or "words" in reason.lower()


def test_question_words_trigger_deep(router):
    """Test that queries with question words route to DeepResearchAgent."""
    question_queries = [
        "How does this work?",
        "Why is this happening?",
        "What is the best approach?",
        "When should I use this?",
        "Where is this configured?"
    ]

    for query in question_queries:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type == AgentType.DEEP, f"Query '{query}' with question word should route to DEEP agent"


def test_logical_operators_trigger_deep(router):
    """Test that queries with logical operators route to DeepResearchAgent."""
    logical_queries = [
        "Redis and Kubernetes integration",
        "Caching or database optimization",
        "Not using Python but JavaScript",
        "API design but with microservices"
    ]

    for query in logical_queries:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type == AgentType.DEEP, f"Query '{query}' with logical operators should route to DEEP agent"


def test_routing_consistency(router):
    """Test that routing is consistent for the same query."""
    query = "How to optimize database queries?"

    results = [router._analyze_query(query, verbose=False) for _ in range(5)]
    agent_types = [agent_type for agent_type, _ in results]

    # All should route to the same agent
    assert all(at == agent_types[0] for at in agent_types), "Routing should be consistent"


def test_router_search_delegates_correctly(router):
    """Test that router.search() delegates to the correct agent."""
    # Note: This test requires ChromaDB/Ollama to be available
    # Skip if not in integration test mode
    try:
        # Simple query should use fast agent
        result = router.search("test query", n_results=1, verbose=False)

        assert result is not None
        assert hasattr(result, 'status')
        assert hasattr(result, 'agent_type')
        assert result.agent_type in ['fast_search', 'deep_research', 'deep_research_web']

    except Exception as e:
        # If ChromaDB/Ollama not available, test passes (we're just checking delegation logic)
        if "ChromaDB" in str(e) or "Ollama" in str(e) or "connection" in str(e).lower():
            pytest.skip(f"Skipping integration test: {e}")
        else:
            raise


def test_edge_case_empty_query(router):
    """Test that empty or whitespace queries are handled."""
    edge_cases = ["", "   ", "\n\t"]

    for query in edge_cases:
        agent_type, reason = router._analyze_query(query, verbose=False)
        # Should still return a valid agent type (likely FAST for simplicity)
        assert agent_type in [AgentType.FAST, AgentType.DEEP]


def test_edge_case_very_long_query(router):
    """Test that very long queries are handled correctly."""
    very_long_query = " ".join(["word"] * 100)  # 100-word query

    agent_type, reason = router._analyze_query(very_long_query, verbose=False)
    assert agent_type == AgentType.DEEP, "Very long queries should route to DEEP agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
