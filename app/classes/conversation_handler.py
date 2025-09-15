import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional

from app.core.database import db
from app.core.logging import logging

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]


class ConversationHandler:
    @staticmethod
    async def add_message(chat_id: str, role: Role, message: str) -> Optional[int]:
        def _add():
            try:
                message_data = {"chat_id": chat_id, "role": role, "message": message}

                result = db.table("messages").insert(message_data).execute()
                return result.data[0]["id"] if result.data else None

            except Exception as e:
                logger.exception(f"Error adding message: {e}")
                raise Exception(f"Error adding message: {e}")

        return await asyncio.to_thread(_add)

    @staticmethod
    async def get_recent_messages(chat_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        def _get():
            try:
                result = (
                    db.table("messages")
                    .select("*")
                    .eq("chat_id", chat_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )

                return list(reversed(result.data)) if result.data else []

            except Exception as e:
                logger.exception(f"Error getting recent messages: {e}")
                return []

        return await asyncio.to_thread(_get)

    @staticmethod
    async def delete_conversation(chat_id: str) -> None:
        def _delete():
            try:
                result = db.table("messages").delete().eq("chat_id", chat_id).execute()

            except Exception as e:
                logger.exception(f"Error deleting conversation: {e}")
                raise Exception(f"Error deleting conversation: {e}")

        await asyncio.to_thread(_delete)

    @staticmethod
    async def delete_message(message_id: str) -> None:
        def _delete():
            try:
                result = db.table("messages").delete().eq("id", message_id).execute()

            except Exception as e:
                logger.exception(f"Error deleting message: {e}")
                raise Exception(f"Error deleting message: {e}")

        await asyncio.to_thread(_delete)

    @staticmethod
    async def prune_conversation(chat_id: str, keep_last: int = 50) -> None:
        def _prune():
            try:
                result = (
                    db.table("messages")
                    .select("id")
                    .eq("chat_id", chat_id)
                    .order("created_at")
                    .execute()
                )

                if result.data and len(result.data) > keep_last:
                    messages_to_delete = result.data[:-keep_last]
                    message_ids = [msg["id"] for msg in messages_to_delete]

                    db.table("messages").delete().in_("id", message_ids).execute()

            except Exception as e:
                logger.exception(f"Error pruning conversation: {e}")
                raise Exception(f"Error pruning conversation: {e}")

        await asyncio.to_thread(_prune)

    @staticmethod
    async def get_messages_for_openai(chat_id: str, limit: int = 50) -> List[Dict[str, str]]:
        messages = await ConversationHandler.get_recent_messages(chat_id, limit)

        return [{"role": msg["role"], "content": msg["message"]} for msg in messages]
