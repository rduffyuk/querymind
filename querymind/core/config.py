"""Configuration management for QueryMind."""

import os
from typing import Optional
from pydantic import BaseModel, Field


class QueryMindConfig(BaseModel):
    """QueryMind configuration."""

    # Service URLs
    chromadb_url: str = Field(default="http://localhost:8000")
    redis_url: str = Field(default="redis://localhost:6379")
    ollama_url: str = Field(default="http://localhost:11434")

    # Vault configuration
    vault_path: str = Field(default="/vault")
    collection_name: str = Field(default="obsidian_vault_mxbai")

    # Router configuration
    router_fast_threshold: int = Field(default=10)

    # Cache configuration
    cache_ttl_query: int = Field(default=3600)  # 1 hour
    cache_ttl_gather: int = Field(default=300)  # 5 minutes

    # External services (optional)
    serper_api_key: Optional[str] = Field(default=None)
    disable_web_search: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")


def load_config() -> QueryMindConfig:
    """Load configuration from environment variables."""
    return QueryMindConfig(
        chromadb_url=os.getenv("CHROMADB_URL", "http://localhost:8000"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        vault_path=os.getenv("VAULT_PATH", "/vault"),
        collection_name=os.getenv("CHROMADB_COLLECTION", "obsidian_vault_mxbai"),
        router_fast_threshold=int(os.getenv("ROUTER_FAST_THRESHOLD", "10")),
        cache_ttl_query=int(os.getenv("CACHE_TTL_QUERY", "3600")),
        cache_ttl_gather=int(os.getenv("CACHE_TTL_GATHER", "300")),
        serper_api_key=os.getenv("SERPER_API_KEY"),
        disable_web_search=os.getenv("DISABLE_WEB_SEARCH", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# Global config instance
config = load_config()
