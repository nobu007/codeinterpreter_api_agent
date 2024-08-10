import datetime
import os
from typing import Any, Dict, List

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
        self._write_to_file(f"## Step {self.step_count}: LLM Start\n\n")
        self._write_to_file(f"**Timestamp:** {self._get_timestamp()}\n\n")
        self._write_to_file("**Prompts:**\n\n")
        for i, prompt in enumerate(prompts, 1):
            self._write_to_file(f"```\nPrompt {i}:\n{prompt}\n```\n\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._write_to_file("**LLM Response:**\n\n")
        for generation in response.generations[0]:
            self._write_to_file(f"```\n{generation.text}\n```\n\n")
        self._write_to_file("---\n\n")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.step_count += 1
        chain_name = serialized.get("name", "Unknown Chain")
        self._write_to_file(f"## Step {self.step_count}: Chain Start - {chain_name}\n\n")
        self._write_to_file(f"**Timestamp:** {self._get_timestamp()}\n\n")
        self._write_to_file("**Inputs:**\n\n")
        self._write_to_file(f"```\n{inputs}\n```\n\n")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._write_to_file("**Outputs:**\n\n")
        self._write_to_file(f"```\n{outputs}\n```\n\n")
        self._write_to_file("---\n\n")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.step_count += 1
        self._write_to_file(f"## Step {self.step_count}: Agent Action\n\n")
        self._write_to_file(f"**Timestamp:** {self._get_timestamp()}\n\n")
        self._write_to_file(f"**Tool:** {action.tool}\n\n")
        self._write_to_file("**Tool Input:**\n\n")
        self._write_to_file(f"```\n{action.tool_input}\n```\n\n")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self._write_to_file("## Agent Finish\n\n")
        self._write_to_file(f"**Timestamp:** {self._get_timestamp()}\n\n")
        self._write_to_file("**Output:**\n\n")
        self._write_to_file(f"```\n{finish.return_values}\n```\n\n")
        self._write_to_file("---\n\n")

    def _write_to_file(self, text: str) -> None:
        self.file.write(text)
        self.file.flush()

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
