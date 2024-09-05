import os
import subprocess

from langchain_core.tools import StructuredTool
from codeinterpreterapi.config import settings

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import ZoltraakInput

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INVOKE_TASKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../invoke_tasks"))


class ZoltraakTools:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.command_log = []
        self.output_files = []

    @classmethod
    def get_tools_zoltraak(cls, ci_params: CodeInterpreterParams) -> None:
        tools_instance = cls(ci_params=ci_params)
        tools = [
            StructuredTool(
                name="zoltraak",
                description="あいまいなリクエストを元にプログラムの設計をしてからプロトタイプのソースコード群を作成します。\n"
                "このツールは基本的な要件を満たす最低限度の品質を持ったコードを生成できます。\n"
                "プログラミングを開始するときは、このツールを最初に実行してください。",
                func=tools_instance.run,
                coroutine=tools_instance.arun,
                args_schema=ZoltraakInput,
            ),
        ]

        return tools

    def run(self, request: str) -> str:
        return self._common_run(request)

    async def arun(self, request: str) -> str:
        return self._common_run(request)

    def _common_run(self, request: str):
        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = []
            args.append('/home/jinno/.pyenv/shims/zoltraak')
            args.append(f"\"{request}\"")
            args.append('-c')
            args.append('dev_func')
            # args.append(f"eval \"$(pyenv init -)\";zoltraak \"{request}\" -c dev_func")
            output_content = subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=settings.WORK_DIR
            )
            print("_common_run output_content=", type(output_content))
            print("_common_run output_content=", output_content)
            self.command_log.append((args, output_content))
            return output_content

        except subprocess.CalledProcessError as e:
            error_message = f"An error occurred: {e}\nOutput: {e.output}"
            print(error_message)
            self.command_log.append((args, error_message))
            return error_message


def test():
    settings.WORK_DIR = "/tmp"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    tools_instance = ZoltraakTools(ci_params=ci_params)
    test_request = "シンプルなpythonのサンプルプログラムを書いてください。テーマはなんでもいいです。"
    result = tools_instance.run(test_request)
    print("result=", result)
    # assert "test output" in result


if __name__ == "__main__":
    test()
