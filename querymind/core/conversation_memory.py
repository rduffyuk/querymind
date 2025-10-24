"""
QueryMind Conversation Memory (Stub Implementation)

Placeholder for conversation memory functionality.
Full implementation tracked in GitHub Issue #2.
"""

from typing import List, Dict, Any, Optional
from querymind.core.logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Stub implementation for conversation memory.

    This is a minimal placeholder to satisfy imports. Full implementation
    with ChromaDB integration and persistent storage is planned for v1.1.
    """

    def __init__(self):
        """Initialize conversation memory."""
        self.messages: List[Dict[str, Any]] = []
        logger.debug("ConversationMemory initialized (stub implementation)")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to conversation memory.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.messages.append(message)
        logger.debug(f"Added message from {role} (stub)")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in conversation."""
        return self.messages

    def save(self, conversation_id: str) -> None:
        """
        Save conversation to persistent storage (stub).

        Args:
            conversation_id: Unique conversation identifier
        """
        logger.debug(f"ConversationMemory.save called for {conversation_id} (stub - no persistence)")
        # TODO: Implement ChromaDB persistence (Issue #2)

    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages.clear()
        logger.debug("ConversationMemory cleared")
