from typing import List, Optional

from codeboxapi import CodeBox  # type: ignore
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import Callbacks
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel


class CodeInterpreterParams(BaseModel):
    codebox: Optional[CodeBox] = None
    llm: Optional[BaseLanguageModel] = None
    llm_fast: Optional[BaseLanguageModel] = None
    llm_smart: Optional[BaseLanguageModel] = None
    llm_local: Optional[BaseLanguageModel] = None
    tools: Optional[List[BaseTool]] = []
    callbacks: Optional[Callbacks] = None
    verbose: Optional[bool] = False
    is_local: Optional[bool] = True
    is_ja: Optional[bool] = True

    @classmethod
    def get_test_params(cls, llm: BaseLanguageModel):
        return CodeInterpreterParams(llm=llm)

    class Config:
        # for CodeBox
        arbitrary_types_allowed = True
