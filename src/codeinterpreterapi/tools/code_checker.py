import os

from langchain_core.tools import StructuredTool

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import FileInput
from codeinterpreterapi.utils.file_util import FileUtil


class CodeChecker:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.code_log = []
        self.input_files = []
        self.output_files = []

    @classmethod
    def get_tools_code_checker(cls, ci_params: CodeInterpreterParams) -> None:
        tools_instance = cls(ci_params=ci_params)
        tools = [
            StructuredTool(
                name="code_checker",
                description="""このツールは作業中の最新コードを取得します。"
                作業を開始するときは最初に最新コードを確認してください。
                ただし、ユーザがコード全体を指定した場合は、ユーザが指定したコードを優先してください。""",
                func=tools_instance._get_latest_code,
                coroutine=tools_instance._aget_latest_code,
                args_schema=FileInput,  # type: ignore
            ),
        ]

        return tools

    def _get_latest_code_common(self, filename=""):
        target_dir = "./"
        target_filename = "main.py"

        if os.path.isdir(filename):
            target_dir = filename
            target_path = os.path.join(target_dir, target_filename)
        elif os.path.isfile(target_filename):
            target_path = target_filename
        else:
            target_path = os.path.join(target_dir, target_filename)
        if os.path.isfile(target_path):
            FileUtil.read_python_file(filename=target_path)
        else:
            # 初期状態または異常時
            return ""

    def _get_latest_code(self, filename=""):
        return self._get_latest_code_common(filename)

    async def _aget_latest_code(self, filename=""):
        return self._get_latest_code_common(filename)


def test():
    settings.WORK_DIR = "/tmp"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    tools_instance = CodeChecker(ci_params=ci_params)
    result = tools_instance._get_latest_code()
    print("result=", result)
    assert "test output" in result


if __name__ == "__main__":
    test()
