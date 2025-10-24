"""
QueryMind - Intelligent RAG with Smart Query Routing

Local-first RAG system that automatically chooses the best search strategy.
"""

__version__ = "0.1.0"
__author__ = "Ryan Duffy"

# Export main API
from querymind.core.search import search
from querymind.agents.router import auto_search

__all__ = ["search", "auto_search", "__version__"]
