import os
import shlex
import subprocess

from langchain_core.tools import StructuredTool

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import BashCommand

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INVOKE_TASKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../invoke_tasks"))


class BashTools:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.command_log = []
        self.input_files = []
        self.output_files = []

    @classmethod
    def get_tools_bash(cls, ci_params: CodeInterpreterParams) -> None:
        tools_instance = cls(ci_params=ci_params)
        tools = [
            StructuredTool(
                name="bash_tool",
                description="bashコマンドを実行します。\n"
                "コードは文字列として一つにまとめて入力してください。\n"
                "この文字列は非常に長くても構いません。\n"
                "コードは改行で始めないでください。\n"
                "例えば、'cat xx_file.txt | grep yy'のように入力します。\n",
                func=tools_instance.run,
                coroutine=tools_instance.arun,
                args_schema=BashCommand,  # type: ignore
            )
        ]

        return tools

    def run(self, command: str) -> str:
        return self._run(command)

    async def arun(self, command: str) -> str:
        return self._arun(command)

    def _run(self, command: str):
        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = shlex.split(command)
            output_content = subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=INVOKE_TASKS_DIR
            )
            print("_run_local output_content=", type(output_content))
            print("_run_local output_content=", output_content)
            self.command_log.append((command, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            error_message = f"An error occurred: {e}\nOutput: {e.output}"
            print(error_message)
            self.command_log.append((command, error_message))
            return error_message

    async def _arun_local(self, command: str) -> str:
        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = shlex.split(command)
            output_content = await subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=INVOKE_TASKS_DIR
            )
            print("_arun_local output_content=", output_content)
            self.command_log.append((command, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            error_message = f"An error occurred: {e}\nOutput: {e.output}"
            print(error_message)
            self.command_log.append((command, error_message))
            return error_message


def test():
    settings.WORK_DIR = "/tmp"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    tools_instance = BashTools(ci_params=ci_params)
    test_code = "echo 'test output'"
    result = tools_instance.run(test_code)
    print("result=", result)
    assert "test output" in result


if __name__ == "__main__":
    test()
