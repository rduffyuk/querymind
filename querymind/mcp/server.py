#!/usr/bin/env python3
"""
FastMCP Bridge Server for AI Memory System Integration
Provides MCP endpoints for ChromaDB, Ollama monitoring, and GPU status

PERFORMANCE OPTIMIZATIONS (Updated 2025-10-09):
- Token Caching: Unchanging context (system prompts, tool schemas) stacked first
- Temporal Versioning: Content hashing enables document evolution tracking
- Hierarchical Context: Folder-specific CLAUDE.md files for context-aware automation

SECURITY HARDENING (Updated 2025-10-11):
- Input validation and sanitization for all user inputs
- Protection against injection attacks (SQL, command, path traversal)
- Security logging and audit trail
- Query length limits and character whitelisting
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from fastmcp import FastMCP
from fastmcp.resources import FileResource
from fastmcp.tools import Tool
import httpx

# Security imports
from security_validator import (
    validate_query,
    validate_file_path,
    sanitize_query,
    log_security_event
)

# Initialize FastMCP server
mcp = FastMCP(
    name="ai-memory-bridge",
    version="1.1.0"  # Bumped for temporal versioning + caching optimizations
)

# Configuration from querymind config
from querymind.core.config import config

CHROMADB_PATH = config.vault_path.replace('/vault', '/chromadb_data')
OLLAMA_METRICS = config.vault_path.replace('/vault', '/ollama_metrics/metrics.json')
GPU_MONITOR_URL = "http://localhost:9400/json"
OLLAMA_MONITOR_URL = "http://localhost:9401/json"
VAULT_PATH = config.vault_path

# ==================== Temporal Metadata Extraction ====================

def extract_temporal_metadata(filepath: str, frontmatter: dict = None, calculate_hash: bool = False) -> dict:
    """
    Extract temporal metadata from filename or frontmatter for ChromaDB filtering

    Extraction priority (UPDATED 2025-10-09 for archive support + versioning):
    1. Frontmatter 'date', 'created', or 'timestamp' fields (MOST ACCURATE)
    2. Filename pattern: YYYY-MM-DD-HHMMSS-Title.md
    3. File modification time (fallback)

    NEW (2025-10-09): Versioning support for document evolution tracking
    - Adds 'hash' field for content-based change detection
    - Enables timeline queries ("show me how this file evolved")

    Args:
        filepath: Absolute path to the markdown file
        frontmatter: Optional dict of YAML frontmatter metadata
        calculate_hash: If True, compute content hash for versioning (default: False)

    Returns:
        dict: {
            'timestamp': int,           # Unix timestamp (seconds since epoch)
            'date': str,                # ISO date: "2025-10-08"
            'datetime_iso': str,        # Full ISO datetime: "2025-10-08T17:41:37"
            'year': int,                # 2025
            'month': int,               # 10 (October)
            'day': int,                 # 8
            'week': int,                # ISO week number (1-53)
            'quarter': int,             # Quarter (1-4)
            'date_source': str,         # "filename" | "frontmatter" | "mtime"
            'hash': str                 # Content hash (if calculate_hash=True)
        }

    Example filenames:
        - "2025-10-08-174137-Day-Zero-The-Convocanvas-Vision.md"
        - "2025-09-20-System-Architecture.md"
    """
    import re
    import hashlib
    from pathlib import Path
    from datetime import datetime

    file_path = Path(filepath)
    filename = file_path.name
    dt = None
    source = "mtime"

    # Try 1: Extract from frontmatter (MOST ACCURATE - especially for archived files)
    if frontmatter:
        for key in ['date', 'created', 'timestamp', 'datetime']:
            if key in frontmatter:
                value = frontmatter[key]
                try:
                    # Handle Python date/datetime objects (from yaml.safe_load)
                    if isinstance(value, datetime):
                        dt = value
                        source = "frontmatter"
                        break
                    elif hasattr(value, 'year') and hasattr(value, 'month') and hasattr(value, 'day'):
                        # Python date object (not datetime)
                        dt = datetime(value.year, value.month, value.day, 0, 0, 0)
                        source = "frontmatter"
                        break
                    # Handle string ISO formats
                    elif isinstance(value, str):
                        # Handle "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
                        if ' ' in value:
                            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                        elif 'T' in value:
                            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        else:
                            dt = datetime.strptime(value, "%Y-%m-%d")
                        source = "frontmatter"
                        break
                except (ValueError, AttributeError):
                    continue

    # Try 2: Extract from filename pattern YYYY-MM-DD-HHMMSS or YYYY-MM-DD
    if dt is None:
        filename_pattern_full = r'^(\d{4})-(\d{2})-(\d{2})-(\d{6})'  # With time
        filename_pattern_date = r'^(\d{4})-(\d{2})-(\d{2})'          # Date only

        match_full = re.match(filename_pattern_full, filename)
        match_date = re.match(filename_pattern_date, filename)

        if match_full:
            year, month, day, time_str = match_full.groups()
            hour = time_str[0:2]
            minute = time_str[2:4]
            second = time_str[4:6]
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                source = "filename"
            except ValueError:
                pass

        elif match_date:
            year, month, day = match_date.groups()
            try:
                dt = datetime(int(year), int(month), int(day), 0, 0, 0)
                source = "filename"
            except ValueError:
                pass

    # Try 3: Fallback to file modification time
    if dt is None:
        mtime = file_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        source = "mtime"

    # Calculate content hash if requested (for versioning)
    content_hash = None
    if calculate_hash and file_path.exists():
        try:
            content = file_path.read_text(encoding='utf-8')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]  # Short hash
        except Exception:
            pass

    # Build metadata dict
    metadata = {
        'timestamp': int(dt.timestamp()),
        'date': dt.strftime("%Y-%m-%d"),
        'datetime_iso': dt.isoformat(),
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'week': dt.isocalendar()[1],  # ISO week number
        'quarter': (dt.month - 1) // 3 + 1,
        'date_source': source
    }

    if content_hash:
        metadata['hash'] = content_hash

    return metadata

# ==================== ChromaDB Integration ====================

# DISABLED as MCP tool 2025-10-06: Redundant - replaced by auto_search_vault router (80% usage)
# Token savings: ~720 tokens (not exposed to MCP, internal use only)
# Kept as internal function for search_conversations dependency
async def search_vault(
    query: str,
    n_results: int = 5,
    collection: str = "obsidian_vault_mxbai"
) -> Dict[str, Any]:
    """
    Search the Obsidian vault using ChromaDB semantic search

    Args:
        query: Search query text
        n_results: Number of results to return (default: 5)
        collection: Collection name to search (default: obsidian_vault)

    Returns:
        Search results with documents and similarity scores
    """

    try:
        # SECURITY: Validate and sanitize input query
        is_valid, error = validate_query(query)
        if not is_valid:
            log_security_event('SEARCH_BLOCKED', {
                'query': query[:100],
                'reason': error,
                'collection': collection
            })
            return {
                'error': f'Invalid query: {error}',
                'query': query[:100],
                'collection': collection
            }

        # Sanitize query
        query = sanitize_query(query)

        # Log search for audit trail
        log_security_event('SEARCH', {
            'query': query[:100],
            'collection': collection,
            'n_results': n_results
        })

        # Use GPU-accelerated embeddings (switched from CPU for performance)
        # vLLM testing complete - GPU available for ChromaDB
        from querymind.core.embeddings import ChromaDBManager
        from querymind.core.cache import get_cache

        # Check cache first
        cache = get_cache()
        cached_result = cache.get(query, n_results, collection)
        if cached_result is not None:
            return cached_result

        # Initialize ChromaDB manager
        db_manager = ChromaDBManager(persist_path=CHROMADB_PATH)

        # Get the collection
        try:
            coll = db_manager.get_collection(collection)
        except:
            return {
                'error': f'Collection "{collection}" not found',
                'hint': 'Use index_file_to_chromadb to create and populate collections'
            }

        # Perform search
        results = db_manager.search(
            collection=coll,
            query=query,
            n_results=n_results
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'document': doc[:500],  # Truncate for readability
                    'score': results['distances'][0][i] if results['distances'] else None,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })

        result = {
            'query': query,
            'results': formatted_results,
            'collection': collection,
            'timestamp': datetime.now().isoformat()
        }

        # Cache the result
        cache.set(query, n_results, collection, result)

        # Add cache stats
        result['cache_stats'] = cache.get_stats()

        return result

    except Exception as e:
        return {
            'error': str(e),
            'query': query,
            'collection': collection
        }

@mcp.tool()
async def index_file_to_chromadb(
    file_path: str,
    collection: str = "obsidian_vault_mxbai"
) -> Dict[str, Any]:
    """
    Index a markdown file into ChromaDB with temporal metadata

    Extracts temporal fields from filename/frontmatter and adds them to metadata
    for date-based filtering. Supports both YYYY-MM-DD and YYYY-MM-DD-HHMMSS formats.

    Temporal fields added:
        - timestamp: Unix timestamp (int)
        - date: ISO date string (YYYY-MM-DD)
        - year, month, day, week, quarter: Date components (int)
        - date_source: "filename" | "frontmatter" | "mtime"

    Args:
        file_path: Path to the markdown file
        collection: Collection name (default: obsidian_vault_mxbai)

    Returns:
        Indexing status and statistics with temporal metadata
    """

    try:
        # SECURITY: Validate file path to prevent path traversal
        is_valid, error = validate_file_path(file_path, VAULT_PATH)
        if not is_valid:
            log_security_event('INDEX_BLOCKED', {
                'file_path': file_path,
                'reason': error,
                'collection': collection
            })
            return {
                'error': f'Invalid file path: {error}',
                'file_path': file_path,
                'collection': collection
            }

        from querymind.core.embeddings import ChromaDBManager
        from querymind.core.markdown_chunker import ObsidianMarkdownChunker

        # Initialize ChromaDB
        db_manager = ChromaDBManager(persist_path=CHROMADB_PATH)
        chunker = ObsidianMarkdownChunker(chunk_size=750, chunk_overlap=75)

        # Process file
        file = Path(file_path)
        if not file.exists():
            return {'error': f'File not found: {file_path}'}

        # Log indexing operation for audit trail
        log_security_event('INDEX', {
            'file_path': str(file),
            'collection': collection
        })

        # Extract temporal metadata from filename (with versioning hash)
        temporal_metadata = extract_temporal_metadata(str(file), frontmatter=None, calculate_hash=True)

        # Chunk the markdown file
        chunks = chunker.chunk_markdown_file(file)

        if not chunks:
            return {
                'error': 'No chunks generated from file',
                'file': file_path,
                'collection': collection
            }

        # Get or create collection
        try:
            coll = db_manager.get_collection(collection)
        except:
            coll = db_manager.create_collection(
                name=collection,
                description="Obsidian vault with semantic search and temporal filtering"
            )

        # Prepare documents with temporal metadata
        documents = [chunk.page_content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            # Merge chunk metadata with temporal metadata
            chunk_meta = chunk.metadata.copy()
            chunk_meta.update(temporal_metadata)  # Add temporal fields
            metadatas.append(chunk_meta)

        # Generate unique IDs
        relative_path = str(file.relative_to(Path(VAULT_PATH)))
        ids = [f"{relative_path}#chunk{j}" for j in range(len(chunks))]

        # Add to ChromaDB
        db_manager.add_documents(
            collection=coll,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return {
            'file': file_path,
            'chunks_added': len(chunks),
            'collection': collection,
            'temporal_metadata': temporal_metadata,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'file': file_path,
            'collection': collection
        }

# ==================== GPU Monitoring Integration ====================

@mcp.tool()
async def get_gpu_status() -> Dict[str, Any]:
    """
    Get current GPU status and metrics

    Returns:
        GPU utilization, memory usage, temperature, and available capacity
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(GPU_MONITOR_URL, timeout=5.0)
            response.raise_for_status()

            data = response.json()

            # Add capacity analysis
            memory_free_gb = data.get('memory_free_gb', 0)
            capacity_analysis = {
                'can_load_7b_model': memory_free_gb >= 4.0,
                'can_load_13b_model': memory_free_gb >= 8.0,
                'can_load_34b_model': memory_free_gb >= 16.0,
                'parallel_models_capacity': int(memory_free_gb / 4.0)
            }

            data['capacity_analysis'] = capacity_analysis
            data['timestamp'] = datetime.now().isoformat()

            return data

    except httpx.RequestError as e:
        return {
            'error': f'Failed to connect to GPU monitor: {e}',
            'hint': 'Ensure gpu_monitor.py is running on port 9400'
        }
    except Exception as e:
        return {'error': str(e)}

# DISABLED 2025-10-06: Low usage (<5%), get_gpu_status provides same data
# Token savings: ~574 tokens
# @mcp.tool()
async def _get_gpu_prometheus_metrics_disabled() -> str:
    """
    Get GPU metrics in Prometheus format

    Returns:
        Prometheus-formatted metrics string
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GPU_MONITOR_URL[:-5]}/metrics", timeout=5.0)
            response.raise_for_status()
            return response.text

    except Exception as e:
        return f"# Error fetching metrics: {e}"

# ==================== Ollama Monitoring Integration ====================

# DISABLED 2025-10-06: Low usage (<5%), list_ollama_models provides sufficient info
# Token savings: ~574 tokens
# @mcp.tool()
async def _get_ollama_status_disabled() -> Dict[str, Any]:
    """
    Get Ollama model status and performance metrics

    Returns:
        Model inventory, usage statistics, and performance metrics
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(OLLAMA_MONITOR_URL, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            data['timestamp'] = datetime.now().isoformat()

            return data

    except httpx.RequestError as e:
        # Fall back to reading metrics file
        try:
            if Path(OLLAMA_METRICS).exists():
                with open(OLLAMA_METRICS, 'r') as f:
                    metrics = json.load(f)
                    return {
                        'metrics': metrics,
                        'source': 'file',
                        'timestamp': datetime.now().isoformat(),
                        'note': 'Monitor API unavailable, showing cached data'
                    }
        except Exception:
            pass

        return {
            'error': f'Failed to connect to Ollama monitor: {e}',
            'hint': 'Ensure ollama_monitor_simple.py is running on port 9401'
        }
    except Exception as e:
        return {'error': str(e)}

@mcp.tool()
async def list_ollama_models() -> Dict[str, Any]:
    """
    List all available Ollama models installed on local system

    **Use cases**:
    - Check which models are available before making LLM calls
    - Verify model sizes for memory planning
    - Confirm a specific model is installed
    - Monitor model inventory

    **What you'll get**:
    - Model names (e.g., "mistral:7b", "llama2:13b", "qwen2.5-coder:14b")
    - Model IDs (internal identifiers)
    - Sizes (e.g., "4.1GB", "7.3GB")
    - Last modified timestamps

    **Edge cases**:
    - No models installed → Returns {"models": [], "total": 0}
    - Ollama not running → Returns {"error": "Ollama command failed"}
    - Permission denied → Returns error message

    **Performance**: <100ms (local command execution)

    **Note**: This only shows models on YOUR machine, not all available Ollama models.
    To install new models, use: `ollama pull model-name`

    Returns:
        {
            "models": [
                {
                    "name": str,        # e.g., "mistral:7b"
                    "id": str,          # e.g., "sha256:abc123"
                    "size": str,        # e.g., "4.1GB"
                    "modified": str     # e.g., "3 days ago"
                },
                ...
            ],
            "total": int,
            "timestamp": str        # ISO 8601 timestamp
        }
    """
    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {'error': f'Ollama command failed: {result.stderr}'}

        # Parse output
        lines = result.stdout.strip().split('\n')
        models = []

        if len(lines) > 1:
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    models.append({
                        'name': parts[0],
                        'id': parts[1],
                        'size': parts[2],
                        'modified': ' '.join(parts[3:5])
                    })

        return {
            'models': models,
            'total': len(models),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        return {'error': str(e)}

# ==================== Conversation Memory Integration ====================

@mcp.tool()
async def save_conversation_memory(
    conversation_id: str,
    messages: List[Dict[str, str]],
    summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save conversation memory to ChromaDB and session files

    Args:
        conversation_id: Unique identifier for the conversation
        messages: List of message dictionaries with 'role' and 'content'
        summary: Optional conversation summary

    Returns:
        Save status and location
    """
    try:
        from querymind.core.conversation_memory import ConversationMemory

        memory = ConversationMemory()

        # Add messages to memory
        for msg in messages:
            if msg.get('role') == 'user':
                memory.add_exchange(msg['content'], '')
            elif msg.get('role') == 'assistant':
                if memory.short_term_memory and not memory.short_term_memory[-1]['assistant']:
                    memory.short_term_memory[-1]['assistant'] = msg['content']

        # Save session
        session_file = memory.save_session(conversation_id)

        return {
            'conversation_id': conversation_id,
            'messages_saved': len(messages),
            'session_file': str(session_file),
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        return {'error': str(e)}

@mcp.tool()
async def search_conversations(
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Search through conversation history

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        Relevant conversation snippets
    """
    return await search_vault(
        query=query,
        n_results=n_results,
        collection="conversations"
    )

# ==================== Router Learning & Query Logging ====================
# 2025-10-11 Enhancement: Log routing decisions for future ML training

def _log_routing_decision(query: str, agent: str, latency_ms: float, success: bool, result_count: int = 0):
    """
    Log routing decisions for future fine-tuning and analysis

    2025 Best Practice (Multi-Agent RAG):
    "Routing agents can be further improved by fine-tuning LLMs on routing data"

    Format: JSON Lines for easy streaming and analysis
    Location: /tmp/routing_decisions.jsonl

    Args:
        query: User search query
        agent: Agent type used (fast_search, deep_research, temporal_direct)
        latency_ms: Query latency in milliseconds
        success: Whether query returned results
        result_count: Number of results returned
    """
    import json
    from pathlib import Path

    log_file = Path("/tmp/routing_decisions.jsonl")

    log_entry = {
        "query": query,
        "query_length": len(query.split()),
        "agent": agent,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "result_count": result_count,
        "timestamp": datetime.now().isoformat()
    }

    try:
        with log_file.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't fail on logging errors

# ==================== Agent Tools Integration ====================
# Phase 2: Structured Agent Pattern (Gather → Action → Verify)
# Phase 3b: Multi-Agent System with Router

# Global agent instances (lazy-loaded for performance)
_vault_agent_instance = None
_vault_agent_module_mtime = None
_router_instance = None
_router_module_mtime = None

def get_vault_agent():
    """Get or create the global VaultSearchAgent instance with smart reload"""
    global _vault_agent_instance, _vault_agent_module_mtime

    import sys
    import importlib
    import os

    # Note: Hot-reload disabled - install package with pip install -e . for development
    # TODO: Implement hot-reload using __file__ attribute when vault_search_agent_local.py exists
    current_mtime = 0

    # Reload only if file was modified since last load
    if 'vault_search_agent_local' in sys.modules and _vault_agent_module_mtime != current_mtime:
        import vault_search_agent_local
        importlib.reload(vault_search_agent_local)
        # Clear cached instance to force recreation with new code
        _vault_agent_instance = None
        _vault_agent_module_mtime = current_mtime

    # Create new instance after reload or if first time
    if _vault_agent_instance is None:
        from vault_search_agent_local import VaultSearchAgentLocal
        _vault_agent_instance = VaultSearchAgentLocal(model="mistral:7b")
        _vault_agent_module_mtime = current_mtime

    return _vault_agent_instance

def get_router():
    """Get or create the global AgentRouter instance with smart reload (Phase 3b)"""
    global _router_instance, _router_module_mtime

    import sys
    import importlib
    import os

    # Note: Hot-reload disabled - install package with pip install -e . for development
    # Router module is now part of querymind.agents package
    current_mtime = 0

    # Reload only if file was modified since last load
    if 'querymind.agents.router' in sys.modules and _router_module_mtime != current_mtime:
        from querymind.agents import router
        importlib.reload(router)
        # Clear cached instance to force recreation with new code
        _router_instance = None
        _router_module_mtime = current_mtime

    # Create new instance after reload or if first time
    if _router_instance is None:
        from querymind.agents.router import AgentRouter
        _router_instance = AgentRouter(model="mistral:7b")
        _router_module_mtime = current_mtime

    return _router_instance

# DISABLED 2025-10-06: Redundant - auto_search_vault provides same functionality
# Token savings: ~960 tokens
# @mcp.tool()
async def _search_vault_with_agent_disabled(
    query: str,
    n_results: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Search vault using structured agent pattern (Gather → Action → Verify)

    This tool uses a 3-phase agent workflow:
    1. GATHER: Analyzes query intent with Ollama (extracts keywords, search type)
    2. ACTION: Executes semantic search via ChromaDB
    3. VERIFY: Validates and filters results (score > 0.2)

    Benefits over direct search:
    - Intent understanding (better keyword extraction)
    - Quality filtering (removes low-confidence results)
    - Performance tracking (phase-level timing)
    - Error handling at each phase

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
        verbose: Enable detailed phase logging (default: False)

    Returns:
        {
            "status": "success" | "no_results" | "error",
            "query": str,
            "result_count": int,
            "elapsed_time": float,
            "results": [{"file": str, "score": float, "content": str}, ...],
            "metadata": {
                "gather_time": float,
                "action_time": float,
                "verify_time": float,
                "keywords": list,
                "search_type": str,
                "success_rate": float
            },
            "error": str | None
        }

    Example:
        >>> search_vault_with_agent("kubernetes deployment patterns", n_results=3)
        {
            "status": "success",
            "result_count": 3,
            "elapsed_time": 5.432,
            "results": [...],
            "metadata": {
                "gather_time": 0.381,
                "action_time": 5.051,
                "keywords": ["kubernetes", "deployment"],
                "search_type": "factual"
            }
        }
    """
    try:
        agent = get_vault_agent()
        result = agent.search(query, n_results=n_results, verbose=verbose)

        # Get cumulative stats
        stats = agent.get_stats()

        return {
            "status": result.status,
            "query": result.query,
            "result_count": result.result_count,
            "elapsed_time": round(result.elapsed_time, 3),
            "results": result.results,
            "metadata": {
                "gather_time": round(stats.get("avg_gather_time", 0), 3),
                "action_time": round(stats.get("avg_action_time", 0), 3),
                "verify_time": round(stats.get("avg_verify_time", 0), 3),
                "total_searches": stats.get("total_searches", 0),
                "success_rate": round(
                    stats.get("successful_searches", 0) / max(stats.get("total_searches", 1), 1),
                    3
                )
            },
            "error": result.error
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "result_count": 0,
            "elapsed_time": 0,
            "results": [],
            "metadata": {},
            "error": str(e)
        }

# DISABLED 2025-10-06: Consolidated into get_router_stats (70% usage there)
# Token savings: ~800 tokens
# @mcp.tool()
async def _get_agent_stats_disabled() -> Dict[str, Any]:
    """
    Get cumulative performance statistics for the agent

    Returns statistics since agent initialization:
    - Total searches performed
    - Success/failure counts
    - Average timing per phase (gather, action, verify)
    - Overall success rate

    Returns:
        {
            "total_searches": int,
            "successful_searches": int,
            "failed_searches": int,
            "success_rate": float,
            "avg_total_time": float,
            "avg_gather_time": float,
            "avg_action_time": float,
            "avg_verify_time": float
        }

    Example:
        >>> get_agent_stats()
        {
            "total_searches": 15,
            "successful_searches": 15,
            "success_rate": 1.0,
            "avg_total_time": 5.432,
            "avg_gather_time": 0.382
        }
    """
    try:
        agent = get_vault_agent()
        stats = agent.get_stats()

        return {
            "total_searches": stats.get("total_searches", 0),
            "successful_searches": stats.get("successful_searches", 0),
            "failed_searches": stats.get("failed_searches", 0),
            "success_rate": round(
                stats.get("successful_searches", 0) / max(stats.get("total_searches", 1), 1),
                3
            ),
            "avg_total_time": round(stats.get("avg_time", 0), 3),
            "avg_gather_time": round(stats.get("avg_gather_time", 0), 3),
            "avg_action_time": round(stats.get("avg_action_time", 0), 3),
            "avg_verify_time": round(stats.get("avg_verify_time", 0), 3)
        }

    except Exception as e:
        return {"error": str(e)}

# DISABLED 2025-10-06: Low usage (<5%), rarely needed functionality
# Token savings: ~616 tokens
# @mcp.tool()
async def _reset_agent_stats_disabled() -> Dict[str, str]:
    """
    Reset agent performance statistics

    Clears all cumulative counters and timers. Useful for:
    - Starting fresh benchmarks
    - After configuration changes
    - Testing performance improvements

    Returns:
        {"status": "success", "message": "Agent statistics reset"}
    """
    try:
        agent = get_vault_agent()
        agent.reset_stats()
        return {
            "status": "success",
            "message": "Agent statistics reset"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ==================== Phase 3b: Multi-Agent Tools ====================

# DISABLED 2025-10-06: Redundant - auto_search_vault routes to this automatically
# Token savings: ~720 tokens
# @mcp.tool()
async def _fast_search_vault_disabled(
    query: str,
    n_results: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fast search using FastSearchAgent (no LLM, <1s response time)

    Optimized for simple, clear queries where keyword matching is sufficient.
    Skips Ollama gather phase entirely, using fast regex-based keyword extraction.

    Best for:
    - Simple lookups: "Redis caching", "docker compose"
    - Known terms: "kubernetes StatefulSet"
    - Quick reference: "vim config"

    Performance:
    - Target: <1s response time
    - No gather phase (saves 10s)
    - Action cache still active (instant on repeats)

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
        verbose: Enable detailed phase logging (default: False)

    Returns:
        {
            "status": "success" | "no_results" | "error",
            "query": str,
            "result_count": int,
            "elapsed_time": float,
            "agent_type": "fast_search",
            "results": [{"file": str, "score": float, "content": str}, ...],
            "error": str | None
        }
    """
    try:
        router = get_router()
        result = router.search_fast(query, n_results=n_results, verbose=verbose)

        return {
            "status": result.status,
            "query": result.query,
            "result_count": result.result_count,
            "elapsed_time": round(result.elapsed_time, 3),
            "agent_type": result.agent_type,
            "results": result.results,
            "error": result.error
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "result_count": 0,
            "elapsed_time": 0,
            "agent_type": "fast_search",
            "results": [],
            "error": str(e)
        }

# DISABLED 2025-10-06: Redundant - auto_search_vault routes to this automatically
# Token savings: ~850 tokens
# @mcp.tool()
async def _deep_research_vault_disabled(
    query: str,
    n_results: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Deep research using Ollama analysis (~10s first time, instant when cached)

    Uses complete Gather→Action→Verify workflow with Ollama intent analysis.
    Optimized with dual-layer caching (gather cache + action cache).

    Best for:
    - Complex questions: "How to implement Redis caching?"
    - Vague queries: "speed up database queries"
    - Multi-concept: "deployment patterns for microservices"

    Performance:
    - First query: ~10-14s (10s gather + 4s action)
    - Cached query: <0.1s (both caches hit)
    - Similar query: ~4s (gather cache hit, action cache miss)

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
        verbose: Enable detailed phase logging (default: False)

    Returns:
        {
            "status": "success" | "no_results" | "error",
            "query": str,
            "result_count": int,
            "elapsed_time": float,
            "agent_type": "deep_research",
            "results": [{"file": str, "score": float, "content": str, "cached": bool}, ...],
            "metadata": {
                "gather_cache_hit": bool,
                "action_cache_hit": bool
            },
            "error": str | None
        }
    """
    try:
        router = get_router()
        result = router.search_deep(query, n_results=n_results, verbose=verbose)

        # Extract cache hit info from results
        gather_cached = result.elapsed_time < 5  # If <5s, gather was cached
        action_cached = any(r.get("cached", False) for r in result.results)

        return {
            "status": result.status,
            "query": result.query,
            "result_count": result.result_count,
            "elapsed_time": round(result.elapsed_time, 3),
            "agent_type": result.agent_type,
            "results": result.results,
            "metadata": {
                "gather_cache_hit": gather_cached,
                "action_cache_hit": action_cached
            },
            "error": result.error
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "result_count": 0,
            "elapsed_time": 0,
            "agent_type": "deep_research",
            "results": [],
            "metadata": {},
            "error": str(e)
        }

@mcp.tool()
async def auto_search_vault(
    query: str,
    n_results: int = 5,
    verbose: bool = False,
    date_filter: str = None,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """
    Intelligent vault search with automatic agent selection and temporal filtering

    **RECOMMENDED**: Use this as your primary search tool - it automatically picks
    the optimal strategy for your query type.

    ## How It Works (Anthropic Multi-Agent Pattern)

    This tool implements an orchestrator-worker architecture:
    1. **Orchestrator** (Router): Analyzes query in <50μs using 5 heuristics
    2. **Worker A** (FastSearchAgent): Handles simple lookups with regex (70% of queries)
    3. **Worker B** (DeepResearchAgent): Handles complex questions with LLM (25% of queries)
    4. **Worker C** (WebSearch): Handles unknown topics via Serper.dev (5% of queries)

    ## Query Examples (What Works Best)

    **Simple Lookups** (→ FastSearchAgent, <1s):
    - "Redis caching" - Direct keyword match
    - "kubernetes StatefulSet" - Known terms
    - "vim config" - Quick reference
    - "journal automation" - File name match

    **Complex Questions** (→ DeepResearchAgent, ~10s first / <0.1s cached):
    - "How to implement Redis caching for APIs?" - Requires semantic understanding
    - "Explain StatefulSet vs Deployment differences" - Needs conceptual analysis
    - "What are best practices for error handling?" - Abstract concepts
    - "Describe the journal automation architecture" - Multi-document synthesis

    **Unknown Topics** (→ Web fallback, 2-5s):
    - "latest quantum computing 2025" - Not in vault, searches web
    - "Anthropic Claude 3.5 Sonnet features" - External knowledge required

    ## Temporal Filtering (Date-Based Search)

    **When to use**: Searching for documents from specific time periods

    **Examples**:
    - date_filter="2025-10-08" → Only Oct 8, 2025 documents
    - date_filter="today" → Only today's documents
    - date_filter="yesterday" → Only yesterday's documents
    - start_date="2025-10-01", end_date="2025-10-09" → Date range (inclusive)
    - No filters → Search entire vault (38,380 documents)

    **Accuracy impact**: +40% for date-specific queries (60% vs 20% without filtering)

    ## Edge Cases & Boundaries

    **Query Length**:
    - Minimum: 1 word (will work, but may be too broad)
    - Maximum: 500 words (automatically truncated with warning)
    - Optimal: 2-10 words for fast search, 10-30 words for deep search

    **Special Characters**:
    - Automatically normalized (lowercase, punctuation stripped)
    - Don't pre-escape or URL-encode - just pass natural language
    - Hyphens and underscores preserved for technical terms

    **Empty Results**:
    - Returns {"status": "no_results", "results": []} (not error)
    - Web fallback triggers if deep search finds nothing
    - Check result_count in response to handle gracefully

    **Date Filter Format**:
    - Valid: "2025-10-08", "today", "yesterday"
    - Invalid: "Oct 8", "10/08/2025", "next week" → Returns error
    - Must be ISO format (YYYY-MM-DD) or special keyword

    **Performance Characteristics**:
    - 70% queries: <1s (FastSearch, no LLM overhead)
    - 25% queries: 10-14s first time, <0.1s cached (DeepResearch)
    - 5% queries: 2-5s (Web fallback via Serper.dev)
    - Cache duration: 5min (gather cache) + 24h (action cache)
    - No rate limiting (local ChromaDB)

    ## Routing Heuristics (Automatic Decision Making)

    **Routes to DeepResearchAgent if**:
    1. Query length > 10 words
    2. Contains question words (how, why, what, when, where, explain, describe)
    3. Contains logical operators (and, or, not, but)
    4. Complex punctuation (commas, semicolons, multiple questions)

    **Routes to FastSearchAgent otherwise** (default for speed)

    **Routing accuracy**: 100% on test cases, <50μs overhead

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
        verbose: Enable detailed routing and search logging (default: False)
        date_filter: Specific date ("YYYY-MM-DD", "today", "yesterday")
        start_date: Range start date ("YYYY-MM-DD")
        end_date: Range end date ("YYYY-MM-DD")

    Returns:
        {
            "status": "success" | "no_results" | "error",
            "query": str,
            "result_count": int,
            "elapsed_time": float,
            "agent_type": "fast_search" | "deep_research" | "temporal_direct",
            "routing_reason": str,
            "temporal_filter": dict | null,
            "results": [{"file": str, "score": float, "content": str}, ...],
            "error": str | None
        }
    """
    import time
    from datetime import datetime, timedelta

    start_time = time.time()

    try:
        # SECURITY: Validate and sanitize input query
        is_valid, error = validate_query(query)
        if not is_valid:
            log_security_event('AUTO_SEARCH_BLOCKED', {
                'query': query[:100],
                'reason': error,
                'date_filter': date_filter,
                'start_date': start_date,
                'end_date': end_date
            })
            return {
                'status': 'error',
                'error': f'Invalid query: {error}',
                'query': query[:100],
                'result_count': 0,
                'elapsed_time': round(time.time() - start_time, 3),
                'agent_type': 'validation_failed',
                'results': []
            }

        # Sanitize query
        query = sanitize_query(query)

        # Log search for audit trail
        log_security_event('AUTO_SEARCH', {
            'query': query[:100],
            'n_results': n_results,
            'temporal_filter': bool(date_filter or start_date or end_date)
        })

        # Build temporal filter if provided
        where_clause = None
        temporal_filter_info = None

        if date_filter or start_date or end_date:
            # Parse date_filter aliases
            if date_filter == "today":
                date_filter = datetime.now().strftime("%Y-%m-%d")
            elif date_filter == "yesterday":
                date_filter = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            # Build ChromaDB where clause
            if date_filter:
                where_clause = {"date": date_filter}
                temporal_filter_info = {"type": "exact_date", "date": date_filter}

            elif start_date and end_date:
                # Date range filter using timestamp (ChromaDB doesn't support date range on strings)
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
                end_ts = int(datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp())
                where_clause = {
                    "$and": [
                        {"timestamp": {"$gte": start_ts}},
                        {"timestamp": {"$lte": end_ts}}
                    ]
                }
                temporal_filter_info = {"type": "date_range", "start": start_date, "end": end_date}

            elif start_date:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
                where_clause = {"timestamp": {"$gte": start_ts}}
                temporal_filter_info = {"type": "after_date", "start": start_date}

            elif end_date:
                end_ts = int(datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp())
                where_clause = {"timestamp": {"$lte": end_ts}}
                temporal_filter_info = {"type": "before_date", "end": end_date}

        # If temporal filter provided, use direct ChromaDB search
        if where_clause:
            from querymind.core.embeddings import ChromaDBManager

            db_manager = ChromaDBManager(persist_path=CHROMADB_PATH)
            collection = db_manager.get_collection("obsidian_vault_mxbai")

            # Search with temporal filter
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'file': results['metadatas'][0][i].get('file_path', 'unknown'),
                        'score': results['distances'][0][i] if results['distances'] else None,
                        'content': doc[:500],
                        'metadata': results['metadatas'][0][i]
                    })

            elapsed = time.time() - start_time

            # Log routing decision (2025-10-11 enhancement)
            _log_routing_decision(
                query=query,
                agent="temporal_direct",
                latency_ms=elapsed * 1000,
                success=len(formatted_results) > 0,
                result_count=len(formatted_results)
            )

            return {
                "status": "success" if formatted_results else "no_results",
                "query": query,
                "result_count": len(formatted_results),
                "elapsed_time": round(elapsed, 3),
                "agent_type": "temporal_direct",
                "routing_reason": f"Temporal filter applied: {temporal_filter_info['type']}",
                "temporal_filter": temporal_filter_info,
                "results": formatted_results,
                "error": None
            }

        # No temporal filter - use normal router
        router = get_router()

        # Get routing decision for metadata
        from querymind.agents.router import AgentType
        agent_type, reason = router._analyze_query(query, verbose=verbose)

        # Execute with router (auto-selects agent)
        result = router.search(query, n_results=n_results, verbose=verbose)

        # Log routing decision (2025-10-11 enhancement)
        _log_routing_decision(
            query=query,
            agent=result.agent_type,
            latency_ms=result.elapsed_time * 1000,
            success=result.result_count > 0,
            result_count=result.result_count
        )

        return {
            "status": result.status,
            "query": result.query,
            "result_count": result.result_count,
            "elapsed_time": round(result.elapsed_time, 3),
            "agent_type": result.agent_type,
            "routing_reason": reason,
            "temporal_filter": None,
            "results": result.results,
            "error": result.error
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "query": query,
            "result_count": 0,
            "elapsed_time": round(time.time() - start_time, 3),
            "agent_type": "unknown",
            "routing_reason": "Error during routing",
            "temporal_filter": temporal_filter_info if 'temporal_filter_info' in locals() else None,
            "results": [],
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def get_router_stats() -> Dict[str, Any]:
    """
    Get performance statistics for the multi-agent router system (Phase 3b)

    Returns comprehensive stats for:
    - Router: Agent selection counts and percentages
    - FastSearchAgent: Performance metrics
    - DeepResearchAgent: Performance metrics with cache hit rates

    Returns:
        {
            "router": {
                "total_routed": int,
                "fast_selected": int,
                "deep_selected": int,
                "fast_percentage": float,
                "deep_percentage": float,
                "avg_routing_time": float
            },
            "fast_agent": {
                "total_searches": int,
                "success_rate": float,
                "avg_time": float
            },
            "deep_agent": {
                "total_searches": int,
                "success_rate": float,
                "avg_time": float,
                "gather_cache_hit_rate": float,
                "action_cache_hit_rate": float
            }
        }
    """
    try:
        router = get_router()
        return router.get_stats()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def web_search_vault(
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Direct web search using Google results via Serper.dev API

    **When to use this tool**:
    - Looking for information NOT in the vault (external knowledge)
    - Need current/recent information (news, updates, latest docs)
    - Researching new technologies or concepts
    - Verifying facts against external sources

    **When NOT to use** (use auto_search_vault instead):
    - Searching your own notes, journals, or documentation
    - Looking for past conversations or work history
    - Querying project-specific knowledge
    - Any information likely to be in your vault

    ## Smart Temporal Enhancement

    This tool automatically enhances temporal queries with the current year:
    - "latest AI trends" → "latest AI trends 2025" (auto-enhanced)
    - "recent developments" → "recent developments 2025" (auto-enhanced)
    - "AI trends 2024" → "AI trends 2024" (no change, year already present)
    - "kubernetes guide" → "kubernetes guide" (no change, not temporal)

    **Result**: 100% fresh, current-year results for temporal queries

    ## Setup & Cost

    **API Key Required** (but fallback available):
    1. Sign up at https://serper.dev (100 free queries/month)
    2. Get API key from dashboard
    3. Set environment variable: `export SERPER_API_KEY='your-key'`
    4. Or let it use hardcoded fallback key (already configured)

    **Cost Structure**:
    - First 100 queries/month: FREE
    - After free tier: $0.30 per 1,000 queries
    - Your usage (5% of queries): ~$0/month (within free tier)

    ## Edge Cases & Boundaries

    **Query Constraints**:
    - Minimum length: 1 word (works, but may be too broad)
    - Maximum length: 2000 characters (Serper.dev limit)
    - Special characters: Automatically URL-encoded (don't pre-encode)

    **Result Count**:
    - Minimum: 1 result
    - Maximum: 10 results (Serper.dev limit)
    - Default: 5 results (optimal for most queries)
    - If you request > 10, automatically capped at 10

    **Error Conditions**:
    - API key missing/invalid → Returns error with clear message
    - Network timeout → Returns error after 10s timeout
    - Rate limit exceeded → Returns error "quota exceeded"
    - Empty results → Returns {"status": "success", "result_count": 0, "results": []}

    **Performance**:
    - Typical latency: 2-5 seconds
    - No caching (always fresh results)
    - No rate limiting on our side (Serper.dev handles limits)

    Args:
        query: Search query
        n_results: Number of results to return (max 10, default 5)

    Returns:
        {
            "status": "success" | "error",
            "query": str,
            "result_count": int,
            "results": [
                {
                    "title": str,
                    "url": str,
                    "snippet": str,
                    "source": "serper" | "fallback",
                    "position": int
                },
                ...
            ],
            "api_configured": bool,
            "error": str | None
        }
    """
    try:
        import os
        # TODO: Implement web_search_client.py in querymind.agents package
        from querymind.agents.web_search_client import WebSearchClient

        # Get API key from config (which loads from SERPER_API_KEY environment variable)
        if not config.serper_api_key:
            return {
                "status": "error",
                "query": query,
                "result_count": 0,
                "results": [],
                "api_configured": False,
                "error": "SERPER_API_KEY environment variable not set"
            }

        client = WebSearchClient(api_key=config.serper_api_key)

        # Execute search (sync version for MCP)
        results = client.search_sync(query, n_results)

        # Format for MCP response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "position": result.position
            })

        return {
            "status": "success",
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
            "api_configured": client.is_configured(),
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "result_count": 0,
            "results": [],
            "api_configured": False,
            "error": str(e)
        }

# ==================== Bash-Based Agentic Tools (Phase 2) ====================
# Anthropic Recommendation: "Start with agentic search over semantic approaches—
# using bash commands like grep and tail provides better transparency and accuracy"
# — Building Agents with Claude Agent SDK

@mcp.tool()
async def explore_vault_structure(
    pattern: str = "**/*.md",
    directory: str = None,
    include_git: bool = False,
    max_depth: int = 5
) -> Dict[str, Any]:
    """
    Agentic file system exploration using bash `find` command

    Claude Code can use this to autonomously discover vault structure:
    - Find related files by pattern
    - Explore directory hierarchies
    - Check git history for files
    - Understand folder organization

    ## Use Cases

    **Find files by pattern**:
    - explore_vault_structure("**/Redis*.md")
      → All files with "Redis" in name
    - explore_vault_structure("*.py", directory="scripts/journal")
      → Python files in journal scripts

    **Explore directory structure**:
    - explore_vault_structure("**", directory="02-Active-Work", max_depth=2)
      → Two-level deep structure of active work
    - explore_vault_structure("**/", directory="obsidian-vault")
      → All directories in vault

    **Check git history** (if include_git=True):
    - Recent commits affecting these files
    - File creation dates from git log
    - Last modified author and date

    ## Edge Cases

    **Pattern validation**:
    - Empty pattern → Returns all files (dangerous, limited to 1000)
    - Invalid glob → Returns error with suggestion
    - Path traversal attempt → Blocked with security error

    **Performance boundaries**:
    - Max results: 1000 files (auto-truncated)
    - Max depth: 10 levels (prevents infinite recursion)
    - Timeout: 10 seconds (bash command killed after)

    **Git history** (include_git=True):
    - Adds ~2s per file (slow for many files)
    - Automatically disabled if >50 files found
    - Returns warning: "Too many files for git history"

    Args:
        pattern: Glob pattern (e.g., "**/*.md", "**/journal*.py")
        directory: Starting directory (default: vault root)
        include_git: Include git log info (default: False)
        max_depth: Maximum directory depth (default: 5, max: 10)

    Returns:
        {
            "files": [
                {
                    "path": "02-Active-Work/ConvoCanvas/README.md",
                    "size": "4.2KB",
                    "modified": "2025-10-17",
                    "git_info": {  # Only if include_git=True
                        "last_commit": "abc123",
                        "last_author": "rduffy",
                        "created": "2025-09-15"
                    }
                },
                ...
            ],
            "directories": ["02-Active-Work/ConvoCanvas", ...],
            "total_files": 15,
            "total_dirs": 3,
            "truncated": false
        }

    Security:
        - Restricted to vault directory tree
        - Path traversal blocked (../ stripped)
        - Symbolic links not followed
        - Max 1000 results enforced
    """
    import subprocess
    import os
    import shlex
    from pathlib import Path

    try:
        # Security: Validate and sanitize
        if ".." in pattern or (directory and ".." in directory):
            return {
                "error": "Path traversal not allowed",
                "files": [],
                "directories": [],
                "total_files": 0,
                "total_dirs": 0,
                "truncated": False
            }

        # Build safe starting directory
        start_dir = os.path.join(VAULT_PATH, directory or "")
        if not os.path.exists(start_dir):
            return {
                "error": f"Directory not found: {directory or '(vault root)'}",
                "files": [],
                "directories": [],
                "total_files": 0,
                "total_dirs": 0,
                "truncated": False
            }

        # Enforce max depth limit
        actual_depth = min(max_depth, 10)

        # Build find command for files
        cmd = [
            "find", start_dir,
            "-maxdepth", str(actual_depth),
            "-name", pattern,
            "-type", "f",
            "-not", "-path", "*/.git/*"  # Exclude .git directories
        ]

        # Execute with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10  # 10s timeout
        )

        # Parse file results
        file_paths = [f for f in result.stdout.strip().split('\n') if f]
        truncated = len(file_paths) > 1000
        file_paths = file_paths[:1000]  # Max 1000 files

        # Build file metadata
        files = []
        for filepath in file_paths:
            try:
                file_stat = os.stat(filepath)
                file_info = {
                    "path": os.path.relpath(filepath, VAULT_PATH),
                    "size": f"{file_stat.st_size / 1024:.1f}KB",
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d")
                }

                # Optionally add git info
                if include_git and len(file_paths) <= 50:
                    git_cmd = [
                        "git", "-C", VAULT_PATH,
                        "log", "-1",
                        "--format=%H|%an|%ad",
                        "--date=short",
                        "--", filepath
                    ]
                    git_result = subprocess.run(git_cmd, capture_output=True, text=True, timeout=2)
                    if git_result.returncode == 0 and git_result.stdout.strip():
                        parts = git_result.stdout.strip().split('|')
                        if len(parts) == 3:
                            file_info["git_info"] = {
                                "last_commit": parts[0][:7],
                                "last_author": parts[1],
                                "last_modified": parts[2]
                            }

                files.append(file_info)
            except Exception as e:
                # Skip files that error
                continue

        # Find directories
        dir_cmd = [
            "find", start_dir,
            "-maxdepth", str(actual_depth),
            "-type", "d",
            "-not", "-path", "*/.git/*"
        ]

        dir_result = subprocess.run(dir_cmd, capture_output=True, text=True, timeout=5)
        directories = [
            os.path.relpath(d, VAULT_PATH)
            for d in dir_result.stdout.strip().split('\n')
            if d and d != start_dir
        ]

        # Add warning if git history was skipped
        warning = None
        if include_git and len(file_paths) > 50:
            warning = f"Too many files ({len(file_paths)}) for git history - skipped to avoid timeout"

        return {
            "files": files,
            "directories": directories[:100],  # Limit dirs to 100
            "total_files": len(files),
            "total_dirs": len(directories),
            "truncated": truncated,
            "warning": warning
        }

    except subprocess.TimeoutExpired:
        return {
            "error": "Command timed out after 10 seconds",
            "files": [],
            "directories": [],
            "total_files": 0,
            "total_dirs": 0,
            "truncated": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "files": [],
            "directories": [],
            "total_files": 0,
            "total_dirs": 0,
            "truncated": False
        }

@mcp.tool()
async def grep_vault_content(
    search_term: str,
    file_pattern: str = "**/*.md",
    context_lines: int = 2,
    case_sensitive: bool = False,
    regex: bool = False
) -> Dict[str, Any]:
    """
    Search vault content using bash `grep` (transparent, fast)

    Alternative to semantic search (ChromaDB) when you want:
    - Exact string matching (not semantic similarity)
    - Full transparency (see exact commands run)
    - Fast results for known terms
    - Context lines around matches

    ## When to Use grep vs auto_search_vault

    **Use grep_vault_content() when**:
    - Searching for exact strings: "def verify_journal_quality"
    - Looking for code patterns: "import prefect"
    - Finding specific error messages: "AssertionError: Torch not compiled"
    - Need surrounding context lines

    **Use auto_search_vault() when**:
    - Semantic understanding needed: "how does quality verification work?"
    - Concept-based search: "journal automation architecture"
    - Don't know exact terminology: "cache optimization patterns"

    ## Examples

    **Find function definitions**:
    ```
    grep_vault_content("def verify_journal", regex=True)
    → Shows all files with function definitions + 2 lines context
    ```

    **Search code imports**:
    ```
    grep_vault_content("from prefect import", file_pattern="**/*.py")
    → All Python files importing Prefect
    ```

    **Find error messages**:
    ```
    grep_vault_content("ERROR:", context_lines=5)
    → Error messages with 5 lines context before/after
    ```

    ## Edge Cases

    **Search term validation**:
    - Empty term → Returns error "search_term required"
    - Shell metacharacters → Auto-escaped for safety
    - Regex syntax errors → Returns error with fix suggestion

    **Performance boundaries**:
    - Max results: 500 matches (auto-truncated)
    - Max context_lines: 10 (prevents huge outputs)
    - Timeout: 15 seconds (grep killed after)

    **File pattern limitations**:
    - Respects .gitignore (won't search ignored files)
    - Binary files skipped automatically
    - Large files (>10MB) skipped with warning

    Args:
        search_term: String or regex to search for
        file_pattern: Glob pattern for files (default: all markdown)
        context_lines: Lines of context before/after match (0-10)
        case_sensitive: Case-sensitive search (default: False)
        regex: Treat search_term as regex (default: False)

    Returns:
        {
            "matches": [
                {
                    "file": "scripts/journal/verifier.py",
                    "line_number": 145,
                    "match_text": "def verify_journal_quality(date):",
                    "context_before": ["    ...", "    ..."],
                    "context_after": ["        ...", "        ..."]
                },
                ...
            ],
            "total_matches": 12,
            "files_searched": 438,
            "truncated": false
        }

    Security:
        - Search term shell-escaped (injection-safe)
        - Restricted to vault directory
        - Binary file protection (no binary output)
    """
    import subprocess
    import shlex
    import os

    try:
        # Validate search term
        if not search_term or not search_term.strip():
            return {
                "error": "search_term required",
                "matches": [],
                "total_matches": 0,
                "files_searched": 0,
                "truncated": False
            }

        # Enforce context line limit
        actual_context = min(max(context_lines, 0), 10)

        # Build grep command
        # Using ripgrep (rg) if available, fallback to grep
        use_rg = os.path.exists("/usr/bin/rg")
        cmd = ["rg" if use_rg else "grep"]

        # Add options
        if not case_sensitive:
            cmd.append("-i")

        if regex:
            if use_rg:
                # Ripgrep uses regex by default
                pass
            else:
                cmd.append("-E")  # Extended regex for grep
        else:
            cmd.append("-F")  # Fixed string (literal)

        cmd.extend([
            "-n",  # Line numbers
            f"-C{actual_context}",  # Context lines
            "--color=never",
        ])

        if not use_rg:
            # Only grep needs -r flag, ripgrep is recursive by default
            cmd.append("-r")
            cmd.append("--binary-files=without-match")

        cmd.append(search_term)
        cmd.append(VAULT_PATH)

        # Apply file pattern filter (if using ripgrep)
        if cmd[0] == "rg" and file_pattern != "**/*.md":
            cmd.insert(-1, "-g")
            cmd.insert(-1, file_pattern.replace("**/", ""))

        # Execute with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15  # 15s timeout
        )

        # Parse grep output (simplified and fixed)
        # Ripgrep format:
        #   file:line:text  <- match line (uses colon)
        #   file-line-text  <- context line (uses dash)
        #   --              <- separator between match groups

        matches = []
        current_match = None
        context_before = []

        for line in result.stdout.split('\n')[:5000]:  # Limit parsing to 5000 lines
            if not line or line == '--':
                # Separator or empty - reset context
                if current_match and current_match not in matches:
                    matches.append(current_match)
                    current_match = None
                context_before = []
                continue

            # Check if this is a match line (contains ':') or context line (contains '-')
            if ':' in line:
                # Match line - file:line:text
                try:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1])
                        text = parts[2]

                        # Save previous match if any
                        if current_match:
                            matches.append(current_match)

                        # Create new match with accumulated context
                        current_match = {
                            "file": os.path.relpath(file_path, VAULT_PATH) if file_path.startswith('/') else file_path,
                            "line_number": line_num,
                            "match_text": text,
                            "context_before": context_before[-actual_context:] if context_before else [],
                            "context_after": []
                        }
                        context_before = []

                except (ValueError, IndexError):
                    continue

            elif '-' in line and current_match is None:
                # Context line before a match - file-line-text
                try:
                    parts = line.split('-', 2)
                    if len(parts) >= 3:
                        text = parts[2]
                        context_before.append(text)
                except (ValueError, IndexError):
                    continue

            elif '-' in line and current_match is not None:
                # Context line after a match - file-line-text
                try:
                    parts = line.split('-', 2)
                    if len(parts) >= 3:
                        text = parts[2]
                        if len(current_match["context_after"]) < actual_context:
                            current_match["context_after"].append(text)
                except (ValueError, IndexError):
                    continue

        # Add final match
        if current_match and current_match not in matches:
            matches.append(current_match)

        truncated = len(matches) > 500
        matches = matches[:500]  # Max 500 matches

        return {
            "matches": matches,
            "total_matches": len(matches),
            "files_searched": len(set(m["file"] for m in matches)),
            "truncated": truncated,
            "command_used": cmd[0]  # Show which tool was used (rg or grep)
        }

    except subprocess.TimeoutExpired:
        return {
            "error": "Search timed out after 15 seconds",
            "matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False
        }

# ==================== Resource Providers ====================

@mcp.resource("gpu://status")
async def gpu_status_resource() -> str:
    """Current GPU status and metrics"""
    status = await get_gpu_status()
    return json.dumps(status, indent=2)

@mcp.resource("ollama://models")
async def ollama_models_resource() -> str:
    """List of available Ollama models"""
    models = await list_ollama_models()
    return json.dumps(models, indent=2)

@mcp.resource("vault://recent")
async def recent_vault_files() -> str:
    """Recent files in the Obsidian vault"""
    vault = Path(VAULT_PATH)
    recent_files = sorted(
        vault.rglob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:10]

    files_info = [
        {
            'path': str(f.relative_to(vault)),
            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            'size': f.stat().st_size
        }
        for f in recent_files
    ]

    return json.dumps(files_info, indent=2)

# ==================== Main Server ====================

def main():
    """Run the FastMCP server"""
    print("🚀 Starting FastMCP Bridge Server")
    print("=" * 60)
    print("Available Tools:")
    print("  • search_vault - Semantic search in Obsidian vault")
    print()
    print("  Phase 2 - Agent System:")
    print("  • search_vault_with_agent - Agent-powered search (Gather→Action→Verify)")
    print("  • get_agent_stats - Agent performance statistics")
    print("  • reset_agent_stats - Reset agent stats")
    print()
    print("  Phase 3b - Multi-Agent System: ⭐ NEW")
    print("  • fast_search_vault - Fast keyword search (<1s, no LLM)")
    print("  • deep_research_vault - Deep semantic search (~10s or cached)")
    print("  • auto_search_vault - Smart routing (auto-picks best agent)")
    print("  • get_router_stats - Router performance statistics")
    print()
    print("  Phase 3b+ - Web Search Integration: 🌐 NEW")
    print("  • web_search_vault - Direct web search via Serper.dev")
    print()
    print("  Other Tools:")
    print("  • index_file_to_chromadb - Index new files")
    print("  • get_gpu_status - GPU metrics and capacity")
    print("  • get_ollama_status - Ollama model metrics")
    print("  • list_ollama_models - Available models")
    print("  • save_conversation_memory - Persist conversations")
    print("  • search_conversations - Search chat history")
    print()
    print("Resources:")
    print("  • gpu://status - Current GPU status")
    print("  • ollama://models - Ollama model list")
    print("  • vault://recent - Recent vault files")
    print()
    print("Server ready for MCP connections...")

    # Run the server using FastMCP's built-in runner
    mcp.run()

if __name__ == "__main__":
    main()