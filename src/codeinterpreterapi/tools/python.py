import base64
import os
import re
import subprocess
import tempfile
from io import BytesIO
from uuid import uuid4

from codeboxapi.schema import CodeBoxOutput  # type: ignore
from langchain_core.tools import StructuredTool

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import CodeInput, File


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
                name="python",
                description="Input a string of code to a ipython interpreter.\n"
                "Write the entire code in a single string.\n"
                "This string can be really long.\n"
                "Do not start your code with a line break.\n"
                "For example, do 'import numpy', not '\\nimport numpy'."
                "Variables are preserved between runs. "
                + (
                    ("You can use all default python packages " f"specifically also these: {settings.CUSTOM_PACKAGES}")
                    if settings.CUSTOM_PACKAGES
                    else ""
                ),  # TODO: or include this in the system message
                func=tools_instance._run_handler,
                coroutine=tools_instance._arun_handler,
                args_schema=CodeInput,  # type: ignore
            ),
        ]

        return tools

    def _store_python_output_file(self, code):
        if settings.PYTHON_OUT_FILE:
            python_file_path = os.path.join(settings.WORK_DIR, settings.PYTHON_OUT_FILE)
            with open(python_file_path, "w", encoding="utf-8") as python_file:
                python_file.write(code)
                return python_file_path
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", dir=settings.WORK_DIR) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
                return temp_file_path
        return None

    def _get_handler_local_command(self, code: str):
        python_file_path = self._store_python_output_file(code)
        command = f"cd src/codeinterpreterapi/invoke_tasks && invoke -c python run-code-file '{python_file_path}'"
        return command

    def _run_handler_local(self, code: str):
        print("_run_handler_local code=", code)
        command = self._get_handler_local_command(code)
        try:
            output_content = subprocess.check_output(command, shell=True, universal_newlines=True)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            return None

    async def _arun_handler_local(self, code: str):
        print("_arun_handler_local code=", code)
        command = self._get_handler_local_command(code)
        try:
            output_content = await subprocess.check_output(command, shell=True, universal_newlines=True)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            return None

    def _run_handler(self, code: str) -> str:
        """Run code in container and send the output to the user"""
        self.show_code(code)
        if self.ci_params.codebox is None:
            return self._run_handler_local(code)
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

    async def _arun_handler(self, code: str) -> str:
        """Run code in container and send the output to the user"""
        await self.ashow_code(code)
        if self.ci_params.codebox is None:
            return self._arun_handler_local(code)
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
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    tools_instance = PythonTools(ci_params=ci_params)
    test_code = "print('test output')"
    result = tools_instance._run_handler(test_code)
    print("result=", result)
    assert "test output" in result


if __name__ == "__main__":
    test()
