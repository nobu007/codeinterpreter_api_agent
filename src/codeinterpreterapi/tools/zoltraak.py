import os
import subprocess

from langchain_core.tools import StructuredTool
from codeinterpreterapi.config import settings

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import ZoltraakInput
from enum import Enum

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INVOKE_TASKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../invoke_tasks"))


class ZoltraakCompilerEnum(Enum):
    PYTHON_CODE = "dev_obj"
    DESIGN = "general_def"
    PROMPT = "general_prompt"


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
                name="zoltraak_python",
                description="あいまいなリクエストを元にプロトタイプのソースコード群を作成します。\n"
                "このツールは基本的な要件を満たす最低限度の品質を持ったコードを生成できます。\n"
                "プログラミングを開始するときは、このツールを最初に実行してください。",
                func=tools_instance.run,
                coroutine=tools_instance.arun,
                args_schema=ZoltraakInput,
            ),
            StructuredTool(
                name="zoltraak_design",
                description="あいまいなリクエストから設計文書を作成します。\n"
                "このツールは基本的な要件を満たすための具体的な設計を定義できます。\n"
                "設計作業を進めるときは、このツールを最初に実行してください。",
                func=tools_instance.run_design,
                coroutine=tools_instance.arun_design,
                args_schema=ZoltraakInput,
            ),
            StructuredTool(
                name="zoltraak_prompt",
                description="あいまいなリクエストから構造化されたユーザリクエスト統一記述書を作成します。\n"
                "リクエストを新規に処理するときは、このツールを最初に実行してください。",
                func=tools_instance.run_prompt,
                coroutine=tools_instance.arun_prompt,
                args_schema=ZoltraakInput,
            ),
        ]

        return tools

    def run(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.PYTHON_CODE.value)

    async def arun(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.PYTHON_CODE.value)

    def run_design(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.DESIGN.value)

    async def arun_design(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.DESIGN.value)

    def run_prompt(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.PROMPT.value)

    async def arun_prompt(self, prompt: str, name: str) -> str:
        return self._common_run(prompt, name, ZoltraakCompilerEnum.PROMPT.value)

    def _common_run(self, prompt: str, name: str, compiler: str):
        # inputのmdファイル名と生成される場所
        input_md_filename = f"{name}_{compiler}.md"
        output_md_path = os.path.abspath(f"pre_{input_md_filename}")

        # promptのmdファイルを生成する
        prompt_md_filename = f"prompt_{input_md_filename}"
        prompt_output_md_path = os.path.abspath(prompt_md_filename)

        # promptをmdファイルに書き込む
        with open(prompt_output_md_path, "w", encoding="utf-8") as file:
            file.write(prompt)

        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = []
            args.append('zoltraak')
            # args.append(f"\"{input_md_filename}\"")
            args.append('-p')
            args.append(f"\"{prompt_md_filename}\"")
            args.append('-c')
            args.append(f"\"{input_md_filename}\"")
            args.append('-ml')
            args.append('1_')
            args.append('-mle')
            args.append('11_')
            if compiler == ZoltraakCompilerEnum.PROMPT.value:
                # プロンプトの場合はzoltraakレイヤ１だけ実行したいのでレガシーモードを指定
                args.append('-mm')
                args.append('zoltraak_legacy')

            print("_common_run command=", " ".join(args))
            output_content = subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=settings.WORK_DIR
            )
            print("_common_run output_content=", type(output_content))
            print("_common_run output_content=", output_content)

            if os.path.isfile(output_md_path):
                with open(output_md_path, "r", encoding="utf-8") as file:
                    output_content = file.read()
                    return output_content
            else:
                print("WARN: no file output_md_path=", output_md_path)
            self.command_log.append((args, output_content))
            print("WARN: no file output_md_path=", output_md_path)
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
    result = tools_instance.run(test_request, "sample")
    print("result=", result)
    # assert "test output" in result


if __name__ == "__main__":
    test()
