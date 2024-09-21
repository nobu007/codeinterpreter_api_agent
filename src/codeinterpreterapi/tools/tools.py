from langchain_community.tools.shell.tool import ShellTool

from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.tools.bash import BashTools
from codeinterpreterapi.tools.code_checker import CodeChecker
from codeinterpreterapi.tools.python import PythonTools
from codeinterpreterapi.tools.zoltraak import ZoltraakTools
from typing import List


class CodeInterpreterTools:
    def __init__(
        self,
        ci_params: CodeInterpreterParams,
    ):
        self._ci_params = ci_params
        self._additional_tools = ci_params.tools

    def get_all_tools(self) -> list[BaseTool]:
        self._additional_tools.extend(PythonTools.get_tools_python(self._ci_params))
        self._additional_tools.extend(CodeChecker.get_tools_code_checker(self._ci_params))
        self.add_tools_shell()
        self.add_tools_web_search()
        self.add_tools_zoltraak()
        return self._additional_tools

    def add_tools_shell(self) -> None:
        use_langchain_shell_tool = True
        if use_langchain_shell_tool:
            # NOT WORKING
            # google.api_core.exceptions.InvalidArgument: 400 * GenerateContentRequest.tools[0].function_declarations[3].parameters.properties[commands].type: must be specified
            # * GenerateContentRequest.tools[0].function_declarations[8].parameters.properties[commands].type: must be specified
            shell_tool = ShellTool()
            shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace("{", "{{").replace(
                "}", "}}"
            )
            tools = [shell_tool]
        else:
            tools = BashTools.get_tools_bash(self._ci_params)
        self._additional_tools += tools

    def add_tools_web_search(self) -> None:
        tools = [TavilySearchResults(max_results=1)]
        self._additional_tools += tools

    def add_tools_zoltraak(self) -> None:
        tools = ZoltraakTools.get_tools_zoltraak(self._ci_params)
        self._additional_tools += tools

    @staticmethod
    def get_zoltraak_tools(ci_params: CodeInterpreterParams) -> None:
        tools = ZoltraakTools.get_tools_zoltraak(ci_params)
        return tools

    @staticmethod
    def get_agent_tools(agent_tools: str, all_tools: List[BaseTool]) -> None:
        selected_tools = []
        for tool in all_tools:
            if tool.name in agent_tools:
                selected_tools.append(tool)
        return selected_tools
