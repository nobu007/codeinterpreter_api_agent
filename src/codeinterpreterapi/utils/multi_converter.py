from typing import Any, Dict, List

from crewai.crews.crew_output import CrewOutput, TaskOutput
from langchain_core.messages import AIMessageChunk


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
        else:
            print("MultiConverter to_str type(input_obj)=", type(input_obj))
            return str(input_obj)

        # 確実にstr以外は念のため再帰
        return MultiConverter.to_str(input_obj)

    @staticmethod
    def _process_ai_message_chunk(chunk: AIMessageChunk) -> str:
        print(f"MultiConverter.to_str AIMessageChunk input_obj= {chunk}")
        if chunk.content:
            return chunk.content
        tool_call_chunks = chunk.tool_call_chunks
        print(f"MultiConverter.to_str AIMessageChunk len= {len(tool_call_chunks)}")
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
