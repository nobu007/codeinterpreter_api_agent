import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain.callbacks import StdOutCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage


class CustomStdOutCallbackHandler(StdOutCallbackHandler):
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201
        print(f"inputs={inputs}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Print out that we are entering a chain."""
        print(f"on_tool_start input_str={input_str}")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        """Run when Chat Model starts running."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} on_chat_model_start...\033[0m")  # noqa: T201


def get_current_function_name(depth: int = 1) -> str:
    return sys._getframe(depth).f_code.co_name


def show_callback_info(name: str, tag: str, data: Any) -> None:
    current_function_name = get_current_function_name(2)
    print("show_callback_info current_function_name=", current_function_name, name)
    print(f"{tag}=", trim_data(data))


def trim_data(data: Union[Any, List[Any], Dict[str, Any]]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :data: 対象データ
    """
    data_copy = deepcopy(data)
    return trim_data_iter("", data_copy)


def trim_data_iter(indent: str, data: Union[Any, List[Any], Dict[str, Any]]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param data: 対象データ
    """
    indent_next = indent + "  "
    if isinstance(data, dict):
        return trim_data_dict(indent_next, data)
    elif isinstance(data, list):
        return trim_data_array(indent_next, data)
    else:
        return trim_data_other(indent, data)


def trim_data_dict(indent: str, data: Dict[str, Any]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    new_data_list = []
    for k, v in data.items():
        new_data_list.append(f"{indent}dict[{k}]: " + trim_data_iter(indent, v))
    return "\n".join(new_data_list)


def trim_data_array(indent: str, data: List[Any]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    new_data_list = []
    for i, item in enumerate(data):
        print(f"{indent}array[{str(i)}]: ")
        new_data_list.append(trim_data_iter(indent, item))
    return "\n".join(new_data_list)


def trim_data_other(indent: str, data: Any) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    stype = str(type(data))
    s = str(data)
    return f"{indent}type={stype}, data={s[:80]}"


class FullOutCallbackHandler(CustomStdOutCallbackHandler):
    # CallbackManagerMixin
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.
        """
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        show_callback_info(class_name, "prompts", prompts)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.
        """
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        show_callback_info(class_name, "messages", messages)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        show_callback_info(class_name, "query", query)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        show_callback_info(class_name, "inputs", inputs)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        show_callback_info(class_name, "input_str", input_str)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        show_callback_info("no_name", "outputs", outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        show_callback_info("no_name", "error", error)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        show_callback_info("no_name", "action", action)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        show_callback_info("no_name", "finish", finish)

    # ToolManagerMixin
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        show_callback_info("no_name", "output", output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        show_callback_info("no_name", "error", error)
