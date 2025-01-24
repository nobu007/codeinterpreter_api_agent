import re
from pathlib import Path
from typing import Callable, List, Union, Dict, Any

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


def assign_runnable_history(runnable: Runnable, runnable_config: RunnableConfig) -> RunnableWithMessageHistory:
    runnable = get_runnable_history(runnable)
    runnable = runnable.with_config(runnable_config)
    return runnable


def get_runnable_history(runnable: Runnable) -> RunnableWithMessageHistory:
    """
    メッセージ履歴を管理するRunnableを生成する。
    異なる入力形式に対応できるよう拡張している。
    """

    class AdaptiveMessageHistory(RunnableWithMessageHistory):
        def _get_input_messages(self, input_val: Dict[str, Any]) -> str:
            """
            異なる入力形式に対応した入力メッセージの取得。
            メッセージ履歴に格納する文字列をinputのdictのkeyから判断する。
            """
            # 型チェック: input_valが辞書形式であることを確認
            if not isinstance(input_val, dict):
                print(
                    f"_get_input_messages Unexpected input type: {type(input_val)}. Expected dict. Input: {input_val}"
                )
                return str(input_val)

            # 標準的な'input'キーをチェック
            if "input" in input_val:
                return self._ensure_string(input_val["input"])

            # AgentのAction形式をチェック
            if "action_input" in input_val:
                action_input = input_val["action_input"]
                # codeキーが存在する場合
                if isinstance(action_input, dict) and "code" in action_input:
                    return self._ensure_string(action_input["code"])
                return self._ensure_string(action_input)

            # その他の既知の入力形式をチェック
            known_keys = ["text", "content", "message"]
            for key in known_keys:
                if key in input_val:
                    return self._ensure_string(input_val[key])

            # 未知のキー形式をログに記録
            print(f"_get_input_messages Unknown input format: {input_val}. Using fallback.")

            # フォールバック: 文字列化して返す
            return str(input_val)

        def _ensure_string(self, value: Any) -> str:
            """
            値を安全に文字列化するヘルパー関数。
            リストの場合は結合し、それ以外は文字列化する。
            """
            if isinstance(value, list):
                try:
                    return ", ".join(map(str, value))  # リストをカンマ区切りの文字列に変換
                except Exception as e:
                    print(f"_ensure_string Failed to join list: {value}. Error: {e}")
                    return str(value)
            return str(value)

    # 拡張されたRunnableWithMessageHistoryを使用
    runnable_with_history = AdaptiveMessageHistory(
        runnable,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="history",
    )

    return runnable_with_history


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# TODO: replace get_by_session_id
def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a factory that can retrieve chat histories.

    The chat histories are keyed by user ID and conversation ID.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A factory that can retrieve chat histories keyed by user ID and conversation ID.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(user_id: str, session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a user id and conversation id."""
        if not _is_valid_identifier(user_id):
            raise ValueError(
                f"User ID {user_id} is not in a valid format. "
                "User ID must only contain alphanumeric characters, "
                "hyphens, and underscores."
                "Please include a valid cookie in the request headers called 'user-id'."
            )
        if not _is_valid_identifier(session_id):
            raise ValueError(
                f"Conversation ID {session_id} is not in a valid format. "
                "Conversation ID must only contain alphanumeric characters, "
                "hyphens, and underscores. Please provide a valid conversation id "
                "via config. For example, "
                "chain.invoke(.., {'configurable': {'session_id': '123'}})"
            )

        user_dir = base_dir_ / user_id
        if not user_dir.exists():
            user_dir.mkdir(parents=True)
        file_path = user_dir / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


def _is_valid_identifier(value: str) -> bool:
    """Check if the value is a valid identifier."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))
