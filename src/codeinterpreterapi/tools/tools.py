from langchain_community.tools.shell.tool import BaseTool, ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import LLM, BaseLanguageModel, BaseLLM
from langchain_core.tools import StructuredTool, Tool
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper
from langchain_experimental.llm_bash.base import LLMBashChain

from codeinterpreterapi.config import settings
from codeinterpreterapi.schema import CodeInput


class CodeInterpreterTools:
    @staticmethod
    def get_all(
        additional_tools: list[BaseTool], llm: BaseLanguageModel, run_handler_func, arun_handler_func
    ) -> list[BaseTool]:
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
        return additional_tools + tools

    @staticmethod
    def get_shell_v2(additional_tools: list[BaseTool], llm: BaseLanguageModel) -> list[BaseTool]:
        """
        ShellTool cause this error. Should not use this.
        pydantic.v1.error_wrappers.ValidationError: 1 validation error for ShellInput
             commands
             field required (type=value_error.missing)
        """
        llm_runnable = ChatWrapper(llm=llm)
        bash_chain = LLMBashChain.from_llm(llm=llm)
        bash_tool = Tool(
            name="Bash", func=bash_chain.invoke, description="Executes bash commands in a terminal environment."
        )

        tools = [bash_tool]
        return additional_tools + tools

    @staticmethod
    def get_web_search(additional_tools: list[BaseTool]) -> list[BaseTool]:
        tools = [TavilySearchResults(max_results=1)]
        return additional_tools + tools
