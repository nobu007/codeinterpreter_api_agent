import os

from langchain.chat_models.base import BaseChatModel

from codeinterpreterapi.config import settings


class CodeInterpreterLlm:
    @classmethod
    def get_llm(cls) -> BaseChatModel:
        model = settings.MODEL
        max_output_tokens = 1024 * 4
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
            )  # type: ignore
        if settings.GEMINI_API_KEY and "gemini" in model:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

            if "gemini" not in model:
                print("Please set the gemini model in the settings.")
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=settings.TEMPERATURE,
                google_api_key=settings.GEMINI_API_KEY,
                max_output_tokens=max_output_tokens,
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
            )
        raise ValueError("Please set the API key for the LLM you want to use.")


def prepare_test_llm():
    model = "gemini-1.5-flash-latest"
    # model = "gemini-1.5-pro-latest"
    # model = "gemini-1.0-pro"
    # model = "claude-3-haiku-20240307"
    # model = "claude-3-sonnet-20240229"
    if "gemini" in model:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_output_tokens=1024 * 4,
        )
    else:
        from langchain_anthropic import ChatAnthropic  # type: ignore

        llm = ChatAnthropic(
            model_name=model,
            temperature=0.1,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=1024 * 4,
        )

    return llm
