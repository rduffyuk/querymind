#!/usr/bin/env python3
"""
Query Cache Implementation for FastMCP Bridge
Provides in-memory LRU caching for frequent ChromaDB queries
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
from threading import Lock
from querymind.core.logging_config import get_logger

logger = get_logger(__name__)

class QueryCache:
    """
    Thread-safe LRU cache for ChromaDB query results.
    Falls back to in-memory caching if Redis is not available.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400):
        """
        Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (default 24 hours)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

        # Try to connect to Redis if available
        self.redis_client = None
        try:
            import redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=1
            )
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis for query caching")
        except Exception:
            self.use_redis = False
            logger.info("Redis not available, using in-memory cache")

    def _generate_key(self, query: str, n_results: int, collection: str) -> str:
        """Generate cache key from query parameters."""
        cache_data = f"{collection}:{query}:{n_results}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def get(self, query: str, n_results: int, collection: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached query result if available and not expired.

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, n_results, collection)

        if self.use_redis and self.redis_client:
            try:
                # Try Redis first
                cached = self.redis_client.get(f"query:{key}")
                if cached:
                    self.hits += 1
                    return json.loads(cached)
            except Exception:
                pass  # Fall back to in-memory

        # In-memory cache
        with self.lock:
            if key in self.cache:
                entry, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return entry
                else:
                    # Expired, remove it
                    del self.cache[key]

        self.misses += 1
        return None

    def set(self, query: str, n_results: int, collection: str, result: Dict[str, Any]):
        """
        Cache a query result.

        Args:
            query: Query text
            n_results: Number of results requested
            collection: Collection name
            result: Query result to cache
        """
        key = self._generate_key(query, n_results, collection)

        # Try Redis first
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    f"query:{key}",
                    self.ttl_seconds,
                    json.dumps(result)
                )
            except Exception:
                pass  # Fall back to in-memory

        # In-memory cache
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            # Add new entry
            self.cache[key] = (result, time.time())
            self.cache.move_to_end(key)

    def clear(self):
        """Clear all cached entries."""
        if self.use_redis and self.redis_client:
            try:
                for key in self.redis_client.scan_iter("query:*"):
                    self.redis_client.delete(key)
            except Exception:
                pass

        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        redis_status = "Not configured"
        if self.use_redis:
            try:
                if self.redis_client.ping():
                    redis_status = "Connected"
            except Exception:
                redis_status = "Connection failed"

        return {
            "cache_type": "Redis" if self.use_redis else "In-memory",
            "redis_status": redis_status,
            "max_size": self.max_size,
            "current_size": len(self.cache),
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "memory_used_mb": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cache in MB."""
        if not self.cache:
            return 0.0

        # Rough estimation
        total_size = 0
        for key, (value, _) in self.cache.items():
            total_size += len(key) + len(json.dumps(value))

        return total_size / (1024 * 1024)


# Singleton instance
_cache_instance = None
_cache_lock = Lock()

def get_cache() -> QueryCache:
    """Get or create singleton cache instance."""
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = QueryCache()

    return _cache_instance


# Integration with FastMCP
def cached_search(search_function):
    """
    Decorator to add caching to search functions.

    Usage:
        @cached_search
        async def search_vault(query, n_results, collection):
            # Original search implementation
            ...
    """
    async def wrapper(query: str, n_results: int = 5, collection: str = "obsidian_vault_mxbai"):
        cache = get_cache()

        # Check cache first
        cached_result = cache.get(query, n_results, collection)
        if cached_result is not None:
            # Add cache hit indicator
            cached_result["_cached"] = True
            cached_result["_cache_stats"] = cache.get_stats()
            return cached_result

        # Not in cache, perform actual search
        result = await search_function(query, n_results, collection)

        # Cache the result
        cache.set(query, n_results, collection, result)

        # Add cache miss indicator
        result["_cached"] = False
        result["_cache_stats"] = cache.get_stats()

        return result

    return wrapper


if __name__ == "__main__":
    """Test the cache implementation."""
    print("🧪 Testing Query Cache Implementation\n")

    cache = get_cache()

    # Test basic operations
    print("Testing basic cache operations...")

    # Test set/get
    test_result = {
        "documents": [["Doc 1", "Doc 2"]],
        "distances": [[0.1, 0.2]],
        "ids": [["id1", "id2"]]
    }

    cache.set("test query", 5, "test_collection", test_result)
    retrieved = cache.get("test query", 5, "test_collection")

    assert retrieved == test_result, "Cache retrieval failed"
    print("✅ Basic set/get working")

    # Test cache miss
    miss = cache.get("non-existent", 5, "test_collection")
    assert miss is None, "Cache miss should return None"
    print("✅ Cache miss handling working")

    # Test stats
    stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Query cache implementation ready for use!")