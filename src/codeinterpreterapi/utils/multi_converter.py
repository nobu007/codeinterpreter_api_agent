from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Protocol, Union

from crewai.crews.crew_output import CrewOutput, TaskOutput
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.agents import AgentFinish
from codeinterpreterapi.schema import CodeInterpreterIntermediateResult, CodeInterpreterPlanList


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict


class MultiConverter:
    @staticmethod
    def to_str(input_obj: Any) -> str:
        if isinstance(input_obj, str):
            return input_obj
        if isinstance(input_obj, AIMessageChunk):
            input_obj = MultiConverter._process_ai_message_chunk(input_obj)
        elif isinstance(input_obj, List):
            if len(input_obj) > 0:
                input_obj = MultiConverter._process_dict(input_obj[-1])
            else:
                return "no output"
        elif isinstance(input_obj, Dict):
            input_obj = MultiConverter._process_dict(input_obj)
        elif isinstance(input_obj, CrewOutput):
            input_obj = MultiConverter._process_crew_output(input_obj)
        elif isinstance(input_obj, ChatPromptValue):
            input_obj = input_obj.to_string()
        elif isinstance(input_obj, AgentFinish):
            input_obj = input_obj.messages()
        elif isinstance(input_obj, CodeInterpreterPlanList):
            input_obj = str(input_obj)
        else:
            print("MultiConverter to_str unknown type(input_obj)=", type(input_obj))
            return str(input_obj)

        # 確実にstr以外は念のため再帰
        return MultiConverter.to_str(input_obj)

    @staticmethod
    def _process_ai_message_chunk(chunk: AIMessageChunk) -> str:
        if chunk.content:
            return chunk.content
        tool_call_chunks = chunk.tool_call_chunks
        if tool_call_chunks:
            last_chunk = tool_call_chunks[-1]
            return last_chunk.get("text", str(last_chunk))
        return str(chunk)

    @staticmethod
    def _process_dict(input_dict: Dict[str, Any]) -> str:
        if "output" in input_dict:
            return input_dict["output"]

        keys = ["tool", "tool_input_obj", "log"]
        code_log_item = {key: str(input_dict[key]) for key in keys if key in input_dict}
        return str(code_log_item) if code_log_item else str(input_dict)

    @staticmethod
    def _process_crew_output(input_crew_output: CrewOutput) -> str:
        # TODO: return json or
        last_task_output: TaskOutput = input_crew_output.tasks_output[-1]
        if last_task_output.json_dict:
            return str(last_task_output.json_dict)
        elif last_task_output.pydantic:
            return str(last_task_output.pydantic)
        else:
            return last_task_output.raw

    @staticmethod
    def _process_crew_output(input_crew_output: CrewOutput) -> str:
        # TODO: return json or
        last_task_output: TaskOutput = input_crew_output.tasks_output[-1]
        if last_task_output.json_dict:
            return str(last_task_output.json_dict)
        elif last_task_output.pydantic:
            return str(last_task_output.pydantic)
        else:
            return last_task_output.raw

    @staticmethod
    def to_CodeInterpreterIntermediateResult(input_dict: Dict) -> Union[CodeInterpreterIntermediateResult, str]:
        """汎用的なレスポンス処理(配列には未対応)"""
        output = CodeInterpreterIntermediateResult(context="")
        is_empty = True

        # CodeInterpreterIntermediateResultの全メンバ名を取得
        output_attributes = MultiConverter.get_attributes(output)

        for output_attribute in output_attributes:
            if output_attribute in input_dict:
                value = input_dict[output_attribute]
                setattr(output, output_attribute, value)
                is_empty = False

        # 取れない場合はstr経由で返して後で変換する
        if is_empty:
            return str(input_dict)

        return output

    @staticmethod
    def get_attributes(output: Union[Dict, BaseModel, DataclassProtocol]):
        attributes = []

        if is_dataclass(output):
            attributes = [field.name for field in fields(output)]
        elif isinstance(output, BaseModel):
            attributes = [
                attr for attr in dir(output) if not callable(getattr(output, attr)) and not attr.startswith("_")
            ]
        elif hasattr(output, "__dict__"):
            attributes = output.__dict__.keys()
        else:
            # default
            attributes = ["content", "thought", "code", "agent_name"]
        return attributes
