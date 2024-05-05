from langchain_community.tools.shell.tool import ShellInput
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool, StructuredTool

from codeinterpreterapi.config import settings
from codeinterpreterapi.schema import CodeInput


class CodeInterpreterTools:
    @staticmethod
    def get_all(additional_tools: list[BaseTool], run_handler_func, arun_handler_func) -> list[BaseTool]:
        additional_tools = CodeInterpreterTools.get_python(additional_tools, run_handler_func, arun_handler_func)
        additional_tools = CodeInterpreterTools.get_shell(additional_tools)
        additional_tools = CodeInterpreterTools.get_web_search(additional_tools)
        return additional_tools

    @staticmethod
    def get_python(additional_tools: list[BaseTool], run_handler_func, arun_handler_func) -> list[BaseTool]:
        return additional_tools + [
            StructuredTool(
                name="python",
                description="Input a string of code to a ipython interpreter. "
                "Write the entire code in a single string. This string can "
                "be really long, so you can use the `;` character to split lines. "
                "Start your code on the same line as the opening quote. "
                "Do not start your code with a line break. "
                "For example, do 'import numpy', not '\\nimport numpy'."
                "Variables are preserved between runs. "
                + (
                    ("You can use all default python packages " f"specifically also these: {settings.CUSTOM_PACKAGES}")
                    if settings.CUSTOM_PACKAGES
                    else ""
                ),  # TODO: or include this in the system message
                func=run_handler_func,
                coroutine=arun_handler_func,
                args_schema=CodeInput,  # type: ignore
            ),
        ]

    @staticmethod
    def get_shell(additional_tools: list[BaseTool]) -> list[BaseTool]:
        # TODO: use ShellInput
        # tools = [ShellInput()]
        return additional_tools

    @staticmethod
    def get_web_search(additional_tools: list[BaseTool]) -> list[BaseTool]:
        # TODO: use ShellInput
        tools = [TavilySearchResults(max_results=1)]
        return additional_tools + tools
