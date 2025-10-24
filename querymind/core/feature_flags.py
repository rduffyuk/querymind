"""
Unleash Feature Flags Integration for Core Module

Provides centralized feature flag management using Unleash for the core module.
Scoped to core module functionality only (not all scripts).

Usage:
    from core.feature_flags import flags

    if flags.is_enabled("core.config_validation"):
        # Validate paths on import
        config.validate_paths()

Environment Variables:
    UNLEASH_URL: Unleash API URL (default: http://localhost:4242/api)
    UNLEASH_API_TOKEN: API token for authentication
    UNLEASH_INSTANCE_ID: Instance identifier (default: hostname)
"""

from UnleashClient import UnleashClient
from typing import Optional, Dict, Any
from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class UnleashConfig:
    """Configuration for Unleash client"""
    url: str = os.getenv("UNLEASH_URL", "http://100.113.76.79:4242/api")
    app_name: str = "leveling-life-core"
    instance_id: str = os.getenv("UNLEASH_INSTANCE_ID", os.getenv("HOSTNAME", "default"))
    api_token: str = os.getenv("UNLEASH_API_TOKEN", "*:development.c60760752745e231a94c0c95c4414faac4f4d9e3238a2d148e64b448")
    refresh_interval: int = 15  # seconds
    environment: str = os.getenv("UNLEASH_ENVIRONMENT", "development")


class FeatureFlags:
    """
    Singleton feature flag client for core module.

    Provides methods to check feature flags and get variants for A/B testing.
    If Unleash is unavailable, returns fallback values (fail-safe).

    Core Module Features:
        core.config_validation: Enable path validation on config import
        core.retry_decorator: Enable automatic retry logic for ChromaDB/LLM calls
        core.verbose_logging: Enable verbose logging in core module
        core.chromadb_singleton: Use singleton pattern for ChromaDB client
        core.llm_singleton: Use singleton pattern for LLM client
        core.strict_mode: Enforce strict type checking and validation
    """

    _instance: Optional['FeatureFlags'] = None
    _client: Optional[UnleashClient] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _initialize(self):
        """Initialize Unleash client (lazy initialization)"""
        if self._initialized:
            return

        try:
            config = UnleashConfig()

            self._client = UnleashClient(
                url=config.url,
                app_name=config.app_name,
                instance_id=config.instance_id,
                custom_headers={'Authorization': config.api_token},
                refresh_interval=config.refresh_interval,
                environment=config.environment,
            )

            self._client.initialize_client()
            self._initialized = True
            logger.info(f"Unleash client initialized: {config.url} (app: {config.app_name})")

        except Exception as e:
            logger.warning(f"Failed to initialize Unleash client: {e}. Using fallback values.")
            self._client = None
            self._initialized = True

    def is_enabled(
        self,
        feature: str,
        context: Optional[Dict[str, Any]] = None,
        fallback: bool = False
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            feature: Feature flag name (e.g., "core.config_validation")
            context: Additional context for activation strategies (user_id, session_id, etc.)
            fallback: Value to return if Unleash is unavailable (default: False)

        Returns:
            bool: True if feature is enabled, False otherwise

        Example:
            >>> if flags.is_enabled("core.strict_mode"):
            ...     validate_types()
        """
        if not self._initialized:
            self._initialize()

        if not self._client:
            logger.debug(f"Unleash unavailable, using fallback for {feature}: {fallback}")
            return fallback

        try:
            return self._client.is_enabled(feature, context, fallback)
        except Exception as e:
            logger.warning(f"Error checking feature {feature}: {e}. Using fallback: {fallback}")
            return fallback

    def get_variant(
        self,
        feature: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get variant for A/B testing.

        Args:
            feature: Feature flag name (e.g., "core.logging_strategy")
            context: Additional context for variant selection

        Returns:
            str: Variant name (e.g., "verbose", "minimal", "disabled")

        Example:
            >>> variant = flags.get_variant("core.logging_strategy")
            >>> if variant == "verbose":
            ...     setup_verbose_logging()
        """
        if not self._initialized:
            self._initialize()

        if not self._client:
            logger.debug(f"Unleash unavailable, returning 'disabled' variant for {feature}")
            return "disabled"

        try:
            variant = self._client.get_variant(feature, context)
            return variant.get("name", "disabled") if variant else "disabled"
        except Exception as e:
            logger.warning(f"Error getting variant for {feature}: {e}. Returning 'disabled'")
            return "disabled"

    def destroy(self):
        """Cleanup Unleash client (for testing/shutdown)"""
        if self._client:
            self._client.destroy()
            self._client = None
            self._initialized = False
            logger.info("Unleash client destroyed")


# Singleton instance - import this from other modules
flags = FeatureFlags()


# Core module feature flag definitions (for documentation)
CORE_FLAGS = {
    "core.config_validation": {
        "description": "Enable path validation when config is imported",
        "type": "toggle",
        "default": False,
        "impact": "High - validates all 18 paths exist on import"
    },
    "core.retry_decorator": {
        "description": "Enable automatic retry logic for ChromaDB/LLM calls",
        "type": "toggle",
        "default": True,
        "impact": "Medium - adds resilience to external calls"
    },
    "core.verbose_logging": {
        "description": "Enable verbose logging in core module",
        "type": "toggle",
        "default": False,
        "impact": "Low - increases log verbosity"
    },
    "core.chromadb_singleton": {
        "description": "Use singleton pattern for ChromaDB client",
        "type": "toggle",
        "default": True,
        "impact": "Medium - prevents multiple ChromaDB instances"
    },
    "core.llm_singleton": {
        "description": "Use singleton pattern for LLM client",
        "type": "toggle",
        "default": True,
        "impact": "Medium - prevents multiple LLM client instances"
    },
    "core.strict_mode": {
        "description": "Enforce strict type checking and validation",
        "type": "toggle",
        "default": False,
        "impact": "High - raises errors on type mismatches"
    },
    "core.logging_strategy": {
        "description": "Logging verbosity level",
        "type": "variant",
        "variants": ["minimal", "standard", "verbose", "debug"],
        "default": "standard",
        "impact": "Low - controls log detail level"
    },
    "core.hybrid_search": {
        "description": "Enable BM25 + Vector ensemble search (hybrid search)",
        "type": "toggle",
        "default": False,
        "impact": "Medium - 30-40% better search recall, adds BM25 overhead"
    }
}


if __name__ == "__main__":
    # Test the feature flags integration
    print("🧪 Testing Unleash Feature Flags Integration\n")

    print("1. Testing is_enabled() with fallback:")
    result = flags.is_enabled("core.config_validation", fallback=False)
    print(f"   core.config_validation: {result}\n")

    print("2. Testing is_enabled() with context:")
    context = {"userId": "test-user", "sessionId": "test-session"}
    result = flags.is_enabled("core.strict_mode", context=context, fallback=False)
    print(f"   core.strict_mode: {result}\n")

    print("3. Testing get_variant():")
    variant = flags.get_variant("core.logging_strategy")
    print(f"   core.logging_strategy variant: {variant}\n")

    print("4. Core module feature flags:")
    for flag_name, flag_info in CORE_FLAGS.items():
        print(f"   {flag_name}")
        print(f"      Type: {flag_info['type']}")
        print(f"      Default: {flag_info['default']}")
        print(f"      Impact: {flag_info['impact']}")
        print()

    print("✅ Feature flags integration ready!")
    print("   - Unleash UI: http://localhost:4242")
    print("   - Default credentials: admin / unleash4all")
