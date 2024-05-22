from langchain_community.tools.shell.tool import BaseTool, ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import StructuredTool

from codeinterpreterapi.config import settings
from codeinterpreterapi.schema import CodeInput


class CodeInterpreterTools:
    def __init__(
        self,
        additional_tools: list[BaseTool],
        run_handler_func: callable,
        arun_handler_func: callable,
        llm: BaseLanguageModel,
    ):
        self._additional_tools = additional_tools
        self._run_handler_func = run_handler_func
        self._arun_handler_func = arun_handler_func
        self._llm = llm

    def get_all_tools(self) -> list[BaseTool]:
        self.add_tools_python()
        self.add_tools_shell()
        self.add_tools_web_search()
        return self._additional_tools

    def add_tools_python(self) -> None:
        tools = [
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
                func=self._run_handler_func,
                coroutine=self._arun_handler_func,
                args_schema=CodeInput,  # type: ignore
            ),
        ]
        self._additional_tools += tools

    def add_tools_shell(self) -> None:
        """
        ShellTool cause this error. Should not use this.
        pydantic.v1.error_wrappers.ValidationError: 1 validation error for ShellInput
             commands
             field required (type=value_error.missing)
        """
        shell_tool = ShellTool()
        shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace("{", "{{").replace(
            "}", "}}"
        )
        tools = [shell_tool]
        self._additional_tools += tools

    def add_tools_web_search(self) -> None:
        tools = [TavilySearchResults(max_results=1)]
        self._additional_tools += tools
