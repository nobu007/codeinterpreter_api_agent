from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from google.ai.generativelanguage_v1beta.types import GenerateContentRequest
from google.generativeai.types.content_types import FunctionDeclarationType  # type: ignore[import]
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_google_genai._common import SafetySettingDict
from langchain_google_genai._function_utils import _ToolConfigDict, _ToolDictLike

from codeinterpreterapi.config import settings


class ChatGoogleGenerativeAIWrapper(ChatGoogleGenerativeAI):
    def _prepare_request(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        tools: Optional[Sequence[_ToolDictLike]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GenerateContentRequest, Dict[str, Any]]:
        num_tokens = self.get_num_tokens_from_messages(messages=messages)
        print("_prepare_request num_tokens=", num_tokens)
        return super()._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
        )


class CodeInterpreterLlm:
    @classmethod
    def get_llm(cls, model: str = settings.MODEL) -> BaseChatModel:
        max_output_tokens = 1024 * 4
        max_retries = 0
        if (
            settings.AZURE_OPENAI_API_KEY
            and settings.AZURE_API_BASE
            and settings.AZURE_API_VERSION
            and settings.AZURE_DEPLOYMENT_NAME
        ):
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                temperature=0.03,
                base_url=settings.AZURE_API_BASE,
                api_version=settings.AZURE_API_VERSION,
                azure_deployment=settings.AZURE_DEPLOYMENT_NAME,
                api_key=settings.AZURE_OPENAI_API_KEY,
                max_retries=settings.MAX_RETRY,
                timeout=settings.REQUEST_TIMEOUT,
                max_tokens=max_output_tokens,
                # max_retries=max_retries,
            )  # type: ignore
        if settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.REQUEST_TIMEOUT,
                temperature=settings.TEMPERATURE,
                max_retries=settings.MAX_RETRY,
                max_tokens=max_output_tokens,
                # max_retries=max_retries,
            )  # type: ignore
        if settings.GEMINI_API_KEY and "gemini" in model:
            if "gemini" not in model:
                print("Please set the gemini model in the settings.")
            return ChatGoogleGenerativeAIWrapper(
                model=model,
                temperature=settings.TEMPERATURE,
                google_api_key=settings.GEMINI_API_KEY,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
            )
        if settings.ANTHROPIC_API_KEY and "claude" in model:
            from langchain_anthropic import ChatAnthropic  # type: ignore

            if "claude" not in model:
                print("Please set the claude model in the settings.")
            return ChatAnthropic(
                model_name=model,
                temperature=settings.TEMPERATURE,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                max_tokens=max_output_tokens,
                max_retries=max_retries,
            )
        raise ValueError("Please set the API key for model=", model)

    @classmethod
    def get_llm_lite(cls, model: str = settings.MODEL_LITE) -> BaseChatModel:
        print("get_llm_lite=", model)
        return cls.get_llm(model=model)

    @classmethod
    def get_llm_fast(cls, model: str = settings.MODEL_FAST) -> BaseChatModel:
        print("get_llm_fast=", model)
        return cls.get_llm(model=model)

    @classmethod
    def get_llm_smart(cls, model: str = settings.MODEL_SMART) -> BaseChatModel:
        print("get_llm_smart=", model)
        return cls.get_llm(model=model)

    @classmethod
    def get_llm_local(cls, model: str = settings.MODEL_LOCAL) -> BaseChatModel:
        print("get_llm_local=", model)
        return cls.get_llm(model=model)

    @classmethod
    def get_llm_switcher(cls, model: str = settings.MODEL_LOCAL) -> Runnable:
        llms = cls.get_llms(model)
        llm_switcher = llms[0]
        fallback_llms = llms[1:]
        llm_switcher = llm_switcher.with_fallbacks(fallback_llms)
        return llm_switcher

    @classmethod
    def get_llm_switcher_tools(cls, model: str = settings.MODEL_LOCAL) -> Runnable:
        llms = cls.get_llms(model)
        llms_tools = []
        for llm in llms:
            if hasattr(llm, "bind_tools"):
                llms_tools.append(llm)

        llm_tools = llms_tools[0]
        fallback_llms = llms_tools[1:]
        llm_tools = llm_tools.with_fallbacks(fallback_llms)
        return llm_tools

    @classmethod
    def get_llms(cls, model: str = settings.MODEL_LOCAL) -> List[BaseChatModel]:
        llms = []
        llms.append(cls.get_llm(model))
        llms.append(cls.get_llm_fast())
        llms.append(cls.get_llm_smart())
        return llms


def prepare_test_llm():
    llm = CodeInterpreterLlm.get_llm_switcher()
    llm_tools = CodeInterpreterLlm.get_llm_switcher_tools()
    runnable_config = RunnableConfig({'callbacks': [StdOutCallbackHandler()]})
    llm = llm.with_config(runnable_config)
    llm_tools = llm_tools.with_config(runnable_config)

    return llm, llm_tools
