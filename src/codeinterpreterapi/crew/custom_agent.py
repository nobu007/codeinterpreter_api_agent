from typing import Any, Dict, List, Optional

from crewai.agents.agent_builder.base_agent import BaseAgent
from langchain_core.tools import BaseTool
from pydantic import Field


class CustomAgent(BaseAgent):
    agent_executor: Any = Field(default=None, description="Verbose mode for the Agent Execution")
    function_calling_llm: Optional[Any] = Field(description="Language model that will run the agent.", default=None)
    allow_code_execution: Optional[bool] = Field(default=False, description="Enable code execution for the agent.")
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )

    def __init__(self, agent_executor: Any, **data):
        config = data.pop("config", {})

        super().__init__(**config, **data)
        self.agent_executor = agent_executor
        self.function_calling_llm = "dummy"  # This is not used
        self.allow_code_execution = False
        self.step_callback = None

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the task description and expected output."""
        print("interpolate_inputs inputs=", inputs)
        super().interpolate_inputs(inputs)

    def execute_task(self, task: Any, context: Optional[str] = None, tools: Optional[List[Any]] = None):
        # AgentExecutorを使用してタスクを実行
        print("execute_task task=", task)
        print("execute_task context=", context)
        input_dict = {}
        input_dict["input"] = task.description
        input_dict["question"] = task.prompt_context
        input_dict["message"] = "タスクを実行してください。\n" + task.expected_output
        result = self.agent_executor.invoke(input=input_dict)
        print("execute_task result=", result)
        return result["output"]

    def create_agent_executor(self, tools=None) -> None:
        pass

    def _parse_tools(self, tools: List[Any]) -> List[Any]:
        return []

    def parse_tools(self, tools: Optional[List[BaseTool]]) -> List[BaseTool]:
        # ツールのパースロジックを実装
        return tools or []

    def get_delegation_tools(self, agents: List[BaseAgent]):
        return []

    def get_output_converter(self, llm, text, model, instructions):
        return lambda x: x  # デフォルトでは変換なし

    def execute(self, task_description: str, context: Optional[List[str]] = None):
        # タスクの実行ロジックを実装
        full_context = "\n".join(context) if context else ""
        full_input = f"{full_context}\n\nTask: {task_description}"
        return self.executor.run(input=full_input)
