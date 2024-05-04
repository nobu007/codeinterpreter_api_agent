from langchain.chat_models.base import BaseChatModel


from codeinterpreterapi.config import settings


class CodeInterpreterLlm:

    @classmethod
    def get_llm(self) -> BaseChatModel:
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
            )  # type: ignore
        if settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.MODEL,
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.REQUEST_TIMEOUT,
                temperature=settings.TEMPERATURE,
                max_retries=settings.MAX_RETRY,
            )  # type: ignore
        if settings.ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic  # type: ignore

            if "claude" not in settings.MODEL:
                print("Please set the claude model in the settings.")
            return ChatAnthropic(
                model_name=settings.MODEL,
                temperature=settings.TEMPERATURE,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
            )
        raise ValueError("Please set the API key for the LLM you want to use.")
