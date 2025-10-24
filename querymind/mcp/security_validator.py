"""
Security validation functions for MCP server

Provides input validation and sanitization to prevent injection attacks.
"""

import re
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def validate_query(query: str, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate search query input

    Args:
        query: User-provided search query
        max_length: Maximum allowed query length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    if len(query) > max_length:
        return False, f"Query too long (max {max_length} characters)"

    # Block common injection patterns
    dangerous_patterns = [
        r';\s*rm\s+-',  # Shell commands
        r';\s*drop\s+',  # SQL injection
        r'<script',      # XSS
        r'\$\(',         # Command substitution
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            log_security_event("dangerous_pattern", query=query[:100])
            return False, "Query contains potentially dangerous pattern"

    return True, None


def validate_file_path(file_path: str, base_path: str = "/vault") -> tuple[bool, Optional[str]]:
    """
    Validate file path to prevent directory traversal

    Args:
        file_path: User-provided file path
        base_path: Base directory that files must be within

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Resolve to absolute path
        resolved = Path(file_path).resolve()
        base = Path(base_path).resolve()

        # Check if path is within base directory
        if not str(resolved).startswith(str(base)):
            log_security_event("path_traversal_attempt", path=file_path)
            return False, "Path outside allowed directory"

        return True, None

    except Exception as e:
        logger.warning(f"Path validation error: {e}")
        return False, f"Invalid path: {str(e)}"


def sanitize_query(query: str) -> str:
    """
    Sanitize query by removing/escaping dangerous characters

    Args:
        query: User-provided query

    Returns:
        Sanitized query string
    """
    # Remove null bytes
    query = query.replace('\x00', '')

    # Remove control characters except newline and tab
    query = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', query)

    # Normalize whitespace
    query = ' '.join(query.split())

    return query.strip()


def log_security_event(event_type: str, **kwargs):
    """
    Log security-related events

    Args:
        event_type: Type of security event
        **kwargs: Additional context
    """
    logger.warning(f"Security event: {event_type}", extra=kwargs)

    # TODO: Implement audit trail to file/database
    # This is a placeholder for production security logging
