import datetime
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks import FileCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class MarkdownFileCallbackHandler(FileCallbackHandler):
    def __init__(self, filename: str = "langchain_log.md"):
        if os.path.isfile(filename):
            os.remove(filename)
        super().__init__(filename, "a")
        self.step_count = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.step_count += 1
        self._write_header("LLM Start")
        self._write_serialized(serialized)
        self._write_to_file("**Prompts:**\n\n")
        for i, prompt in enumerate(prompts, 1):
            self._write_to_file(f"```\nPrompt {i}:\n{prompt}\n```\n\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._write_header("LLM End")
        if response.llm_output:
            self._write_to_file(response.llm_output)
        for generation in response.generations[0]:
            self._write_to_file(f"```\n{generation.text}\n```\n\n")
        self._write_to_file("---\n\n")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.step_count += 1
        chain_name = serialized.get("name", "Unknown Chain")
        self._write_header(f"Chain Start - {chain_name}")
        self._write_serialized(serialized)
        self._write_inputs(inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._write_header("Chain End")
        self._write_outputs(outputs)

    def on_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any) -> Any:
        self.step_count += 1
        self._write_header("Agent Action")
        self._write_to_file(f"**Tool:** {action.tool}\n\n")
        self._write_to_file("**Tool Input:**\n\n")
        self._write_to_file(f"```\n{action.tool_input}\n```\n\n")

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        self._write_header("Agent Finish")
        self._write_to_file("**Output:**\n\n")
        self._write_to_file(f"```\n{finish.return_values}\n```\n\n")
        self._write_to_file("---\n\n")

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
        self.step_count += 1
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self._write_header(f"Tool Start - {class_name}")
        self._write_serialized(serialized)
        self._write_inputs(inputs)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._write_header("Tool End")
        self._write_to_file(f"**Output:**{output}\n\n")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._write_header("Tool Error")
        self._write_to_file(f"{str(error)}\n\n")

    def _write_to_file(self, text: str) -> None:
        self.file.write(text)
        self.file.flush()

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_header(self, title: str):
        self._write_to_file(f"## Step {self.step_count}: {title}\n\n")
        self._write_to_file(f"**Timestamp:** {self._get_timestamp()}\n\n")

    def _write_serialized(self, serialized: Dict[str, Any]):
        name = serialized.get("name", "Unknown")
        self._write_to_file(f"**serialized.name:** {name}\n\n")
        self._write_to_file(f"**serialized.keys:** {serialized.keys()}\n\n")

    def _write_inputs(self, inputs: Dict[str, Any]):
        self._write_to_file("**Inputs:**\n\n")
        self._write_to_file(f"```\n{inputs}\n```\n\n")

    def _write_outputs(self, outputs: Dict[str, Any]):
        self._write_to_file("**Outputs:**\n\n")
        self._write_to_file(f"```\n{outputs}\n```\n\n")
