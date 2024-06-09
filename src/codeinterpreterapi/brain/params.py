from typing import List, Optional

from codeboxapi import CodeBox  # type: ignore
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import Callbacks
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import tool


@tool
def test_plus(first: str, second: str) -> str:
    """Plus two integers together."""
    print("test_plus f=", first, ", s=", second)
    return str(int(first) + int(second))


@tool
def test_multiply(first: str, second: str) -> str:
    """Multiply two integers together."""
    print("test_multiply f=", first, ", s=", second)
    return str(int(first) * int(second))


class CodeInterpreterParams(BaseModel):
    codebox: Optional[CodeBox] = None
    llm_lite: Optional[Runnable] = None
    llm_fast: Optional[Runnable] = None
    llm_smart: Optional[Runnable] = None
    llm_local: Optional[Runnable] = None
    llm: Optional[Runnable] = None
    llm_tools: Optional[Runnable] = None
    tools: Optional[List[BaseTool]] = []
    callbacks: Optional[Callbacks] = None
    verbose: Optional[bool] = False
    is_local: Optional[bool] = True
    is_ja: Optional[bool] = True

    @classmethod
    def get_test_params(cls, llm: BaseLanguageModel, llm_tools: BaseChatModel = None):
        tools = [test_plus, test_multiply]
        return CodeInterpreterParams(
            llm_lite=llm,
            llm_fast=llm,
            llm_smart=llm,
            llm_local=llm,
            llm=llm,
            llm_tools=llm_tools,
            tools=tools,
            verbose=True,
        )

    class Config:
        # for CodeBox
        arbitrary_types_allowed = True
