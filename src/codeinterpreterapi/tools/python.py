import base64
import os
import re
import shlex
import subprocess
from io import BytesIO
from uuid import uuid4

from codeboxapi.schema import CodeBoxOutput  # type: ignore
from langchain_core.tools import StructuredTool

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import CodeAndFileInput, File, FileInput
from codeinterpreterapi.utils.file_util import FileUtil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INVOKE_TASKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../invoke_tasks"))


class PythonTools:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.code_log = []
        self.input_files = []
        self.output_files = []

    @classmethod
    def get_tools_python(cls, ci_params: CodeInterpreterParams) -> None:
        tools_instance = cls(ci_params=ci_params)
        tools = [
            StructuredTool(
                name="python_by_code",
                description="IPythonインタプリタにコードを入力します。\n"
                "コードは文字列として一つにまとめて入力してください。\n"
                "この文字列は非常に長くても構いません。\n"
                "コードは改行で始めないでください。\n"
                "例えば、'import numpy'のように入力します。\n"
                "'\\nimport numpy'のようにはしないでください。\n"
                "また、必ず'filename'を指定してください。この値はファイル名として使用されます。\n"
                "変数は実行間で保持されます。\n"
                "デフォルトのPythonパッケージに加えて、次のカスタムパッケージも使用できます: "
                f"{settings.CUSTOM_PACKAGES}"
                if settings.CUSTOM_PACKAGES
                else "",
                func=tools_instance.run_by_code,
                coroutine=tools_instance.arun_by_code,
                args_schema=CodeAndFileInput,  # type: ignore
            ),
            StructuredTool(
                name="python_by_file",
                description="IPythonインタプリタにpyファイルを入力します。\n"
                "ファイルはカレントディレクトリからの相対パスか絶対パスを指定してください。\n"
                "デフォルトのPythonパッケージに加えて、次のカスタムパッケージも使用できます: "
                f"{settings.CUSTOM_PACKAGES}"
                if settings.CUSTOM_PACKAGES
                else "",
                func=tools_instance.run_by_file,
                coroutine=tools_instance.arun_by_file,
                args_schema=FileInput,  # type: ignore
            ),
        ]

        return tools

    def run_by_file(self, filename: str) -> str:
        return self._run_local(filename)

    def run_by_code(self, filename: str, code: str) -> str:
        return self._run_local(filename, code)

    async def arun_by_file(self, filename: str) -> str:
        return self._arun_local(filename)

    async def arun_by_code(self, filename: str, code: str) -> str:
        return self._arun_local(filename, code)

    def _get_command_local(self, filename: str, code: str = "") -> str:
        python_file_path = FileUtil.get_python_file_path(filename=filename)
        if code:
            python_file_path = FileUtil.write_python_file(filename, code)
        command = f"invoke -c python run-code-file '{python_file_path}'"
        return command

    def _run_local(self, filename: str, code: str = ""):
        command = self._get_command_local(filename, code)
        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = shlex.split(command)
            output_content = subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=INVOKE_TASKS_DIR
            )
            print("_run_local output_content=", type(output_content))
            print("_run_local output_content=", output_content)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            error_message = f"An error occurred: {e}\nOutput: {e.output}"
            print(error_message)
            self.code_log.append((code, error_message))
            return error_message

    async def _arun_local(self, filename: str, code: str = "") -> str:
        print(f"_arun_handler_local filename={filename}, code={code}")
        command = self._get_command_local(filename, code)
        try:
            # シェルインジェクションを防ぐためにshlexを使用
            args = shlex.split(command)
            output_content = await subprocess.check_output(
                args, stderr=subprocess.STDOUT, universal_newlines=True, cwd=INVOKE_TASKS_DIR
            )
            print("_arun_local output_content=", output_content)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            error_message = f"An error occurred: {e}\nOutput: {e.output}"
            print(error_message)
            self.code_log.append((code, error_message))
            return error_message

    def _run_handler(self, filename: str, code: str) -> str:
        """Run code in container and send the output to the user"""
        if self.ci_params.codebox is None:
            return self._run_local(filename, code)
        output: CodeBoxOutput = self.ci_params.codebox.run(code)
        self.code_log.append((code, output.content))

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'",
                    output.content,
                ):
                    self.ci_params.codebox.install(package.group(1))
                    return f"{package.group(1)} was missing but got installed now. Please try again."
            else:
                # TODO: pre-analyze error to optimize next code generation
                pass
            if self.ci_params.verbose:
                print("Error:", output.content)

        # elif modifications := get_file_modifications(code, self.llm):
        #     for filename in modifications:
        #         if filename in [file.name for file in self.input_files]:
        #             continue
        #         fileb = self.codebox.download(filename)
        #         if not fileb.content:
        #             continue
        #         file_buffer = BytesIO(fileb.content)
        #         file_buffer.name = filename
        #         self.output_files.append(File(name=filename, content=file_buffer.read()))

        return output.content

    async def _arun(self, filename: str, code: str) -> str:
        """Run code in container and send the output to the user"""
        await self.ashow_code(code)
        if self.ci_params.codebox is None:
            return self._arun_local(code)
        output: CodeBoxOutput = await self.ci_params.codebox.arun(code)
        self.code_log.append((code, output.content))

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'",
                    output.content,
                ):
                    await self.ci_params.codebox.ainstall(package.group(1))
                    return f"{package.group(1)} was missing but got installed now. Please try again."
            else:
                # TODO: pre-analyze error to optimize next code generation
                pass
            if self.ci_params.verbose:
                print("Error:", output.content)

        # elif modifications := await aget_file_modifications(code):
        #     for filename in modifications:
        #         if filename in [file.name for file in self.input_files]:
        #             continue
        #         fileb = await self.codebox.adownload(filename)
        #         if not fileb.content:
        #             continue
        #         file_buffer = BytesIO(fileb.content)
        #         file_buffer.name = filename
        #         self.output_files.append(File(name=filename, content=file_buffer.read()))

        return output.content

    def show_code(self, code: str) -> None:
        if self.ci_params.verbose:
            print(code)

    async def ashow_code(self, code: str) -> None:
        """Callback function to show code to the user."""
        if self.ci_params.verbose:
            print(code)


def test():
    settings.WORK_DIR = "/tmp"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    tools_instance = PythonTools(ci_params=ci_params)
    test_code = "print('test output')"
    result = tools_instance.run_by_code("main.py", test_code)
    result = tools_instance.run_by_file("main.py")
    print("result=", result)
    assert "test output" in result


if __name__ == "__main__":
    test()
