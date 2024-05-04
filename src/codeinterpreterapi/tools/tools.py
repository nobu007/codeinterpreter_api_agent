from langchain_core.tools import BaseTool, StructuredTool
from codeinterpreterapi.config import settings
from codeinterpreterapi.schema import CodeInput


class CodeInterpreterTools:

    @staticmethod
    def get_tools(
        additional_tools: list[BaseTool], run_handler_func, arun_handler_func
    ) -> list[BaseTool]:
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
                    (
                        "You can use all default python packages "
                        f"specifically also these: {settings.CUSTOM_PACKAGES}"
                    )
                    if settings.CUSTOM_PACKAGES
                    else ""
                ),  # TODO: or include this in the system message
                func=run_handler_func,
                coroutine=arun_handler_func,
                args_schema=CodeInput,  # type: ignore
            ),
        ]
