from typing import Any, Dict, List, Optional

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task as CrewTask
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool
from pydantic import Field

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.utils.multi_converter import MultiConverter


class CustomAgent(BaseAgent):
    agent_executor: AgentExecutor = Field(default=None, description="Verbose mode for the Agent Execution")
    ci_params: CodeInterpreterParams = Field(
        default=None,
        description="All settings for agents and llms.",
    )
    function_calling_llm: Optional[Any] = Field(description="Language model that will run the agent.", default=None)
    allow_code_execution: Optional[bool] = Field(default=False, description="Enable code execution for the agent.")
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )

    def __init__(self, agent_executor: Any, ci_params: CodeInterpreterParams, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)
        self.ci_params = ci_params
        self.agent_executor = agent_executor
        self.function_calling_llm = "dummy"  # This is not used
        self.allow_code_execution = False
        self.step_callback = None

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the task description and expected output."""
        super().interpolate_inputs(inputs)

    def execute_task(self, task: CrewTask, context: Optional[str] = None, tools: Optional[List[Any]] = None) -> str:
        # Notice: ValidationError  - 1 validation error for TaskOutput | raw: Input should be a valid string
        # crewaiのTaskOutputのrawに入るのでstrで返す必要がある。
        # TODO: 直接dictを返せるようにcrewaiを直す？

        # AgentExecutorを使用してタスクを実行
        input_dict = self.create_input_dict(task)
        result = self.agent_executor.invoke(input=input_dict, config=self.ci_params.runnable_config)
        result_str = MultiConverter.to_str(result)

        # TODO: return full dict when crewai is updated
        return result_str

    def create_input_dict(self, task: CrewTask) -> None:
        # This is interface crewai <=> langchain
        # Tools will be set by langchain layer.
        task_description = task.description
        if task.context:
            task_description += f"\n\n### コンテキスト\n{task.context}"
        if task.expected_output:
            task_description += f"\n\n### 出力形式\n{task.expected_output}"

        additional_keys = ["human_input", "goal", "backstory"]

        for key in additional_keys:
            if hasattr(task, key):
                value = getattr(task, key)
                task_description += f"\n\n### {key}\n{value}"
        input_dict = {}
        input_dict["task_description"] = task_description
        return input_dict

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
        print("get_output_converter llm=", type(llm))
        print("get_output_converter text=", type(text))
        print("get_output_converter model=", type(model))
        print("get_output_converter instructions=", type(instructions))
        return lambda x: x  # デフォルトでは変換なし

    def execute(self, task_description: str, context: Optional[List[str]] = None):
        # タスクの実行ロジックを実装
        full_context = "\n".join(context) if context else ""
        full_input = f"{full_context}\n\nTask: {task_description}"
        return self.executor.run(input=full_input)
