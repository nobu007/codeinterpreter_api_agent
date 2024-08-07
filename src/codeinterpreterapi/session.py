import re
import traceback
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Type, Union
from uuid import UUID

from codeboxapi import CodeBox  # type: ignore
from langchain.callbacks.base import BaseCallbackHandler, Callbacks
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from codeinterpreterapi.brain.brain import CodeInterpreterBrain
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.chains import aremove_download_link
from codeinterpreterapi.chat_history import CodeBoxChatMessageHistory
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import CodeInterpreterLlm
from codeinterpreterapi.schema import CodeInterpreterResponse, File, SessionStatus, UserRequest


def _handle_deprecated_kwargs(kwargs: dict) -> None:
    settings.MODEL = kwargs.get("model", settings.MODEL)
    settings.MAX_RETRY = kwargs.get("max_retry", settings.MAX_RETRY)
    settings.TEMPERATURE = kwargs.get("temperature", settings.TEMPERATURE)
    settings.OPENAI_API_KEY = kwargs.get("openai_api_key", settings.OPENAI_API_KEY)
    settings.SYSTEM_MESSAGE = kwargs.get("system_message", settings.SYSTEM_MESSAGE)
    settings.MAX_ITERATIONS = kwargs.get("max_iterations", settings.MAX_ITERATIONS)


class AgentCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, agent_callback_func: callable):
        self.agent_callback_func = agent_callback_func

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        print("AgentCallbackHandler on_chain_start")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        print("AgentCallbackHandler on_chain_end")
        self.agent_callback_func(outputs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        print("AgentCallbackHandler on_chat_model_start")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        print("AgentCallbackHandler on_chain_error")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        print("AgentCallbackHandler on_agent_action")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        print("AgentCallbackHandler on_agent_finish")


class CodeInterpreterSession:
    def __init__(
        self,
        additional_tools: list[BaseTool] = None,
        callbacks: Callbacks = None,
        is_local: bool = True,
        is_ja: bool = True,
        **kwargs: Any,
    ) -> None:
        if additional_tools is None:
            additional_tools = []
        _handle_deprecated_kwargs(kwargs)
        self.verbose = kwargs.get("verbose", settings.DEBUG)
        llm_lite: BaseLanguageModel = CodeInterpreterLlm.get_llm_lite()
        llm_fast: BaseLanguageModel = CodeInterpreterLlm.get_llm_fast()
        llm_smart: BaseLanguageModel = CodeInterpreterLlm.get_llm_smart()
        llm_local: BaseLanguageModel = CodeInterpreterLlm.get_llm_local()
        llm: Runnable = CodeInterpreterLlm.get_llm_switcher()
        llm_tools: Runnable = CodeInterpreterLlm.get_llm_switcher_tools()

        # runnable_config
        init_session_id = "12345678-1234-1234-1234-123456789abc"
        configurable = {"session_id": init_session_id}  # TODO: set session_id
        runnable_config = RunnableConfig(
            configurable=configurable, callbacks=[AgentCallbackHandler(self._output_handler)]
        )

        # ci_params = {}
        self.ci_params = CodeInterpreterParams(
            llm_lite=llm_lite,
            llm_fast=llm_fast,
            llm_smart=llm_smart,
            llm_local=llm_local,
            llm=llm,
            llm_tools=llm_tools,
            tools=additional_tools,
            callbacks=callbacks,
            verbose=self.verbose,
            is_local=is_local,
            is_ja=is_ja,
            runnable_config=runnable_config,
        )
        self.ci_params.session_id = UUID(init_session_id)
        self.brain = CodeInterpreterBrain(self.ci_params)
        self.log("llm=" + str(llm))

        self.input_files: list[File] = []
        self.output_files: list[File] = []
        self.code_log: list[tuple[str, str]] = []

    @classmethod
    def from_id(cls, session_id: UUID, **kwargs: Any) -> "CodeInterpreterSession":
        session = cls(**kwargs)
        session.ci_params.codebox = CodeBox.from_id(session_id)
        session.ci_params.session_id = session_id
        return session

    @property
    def session_id(self) -> Optional[UUID]:
        return self.ci_params.session_id

    def start(self) -> SessionStatus:
        print("start")
        status = SessionStatus.from_codebox_status(self.ci_params.codebox.start())
        self.brain.initialize()
        self.ci_params.codebox.run(
            f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
        )
        return status

    async def astart(self) -> SessionStatus:
        print("astart")
        status = SessionStatus.from_codebox_status(await self.ci_params.codebox.astart())
        self.brain.initialize()
        await self.ci_params.codebox.arun(
            f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
        )
        return status

    def start_local(self) -> SessionStatus:
        print("start_local")
        self.brain.initialize()
        status = SessionStatus(status="started")
        return status

    async def astart_local(self) -> SessionStatus:
        print("astart_local")
        status = self.start_local()
        self.brain.initialize()
        return status

    def _history_backend(self) -> BaseChatMessageHistory:
        return (
            CodeBoxChatMessageHistory(codebox=self.ci_params.codebox)
            if settings.HISTORY_BACKEND == "codebox"
            else (
                RedisChatMessageHistory(
                    session_id=str(self.session_id),
                    url=settings.REDIS_URL,
                )
                if settings.HISTORY_BACKEND == "redis"
                else (
                    PostgresChatMessageHistory(
                        session_id=str(self.session_id),
                        connection_string=settings.POSTGRES_URL,
                    )
                    if settings.HISTORY_BACKEND == "postgres"
                    else ChatMessageHistory()
                )
            )
        )

    def _input_handler(self, request: UserRequest) -> None:
        """Callback function to handle user input."""
        if not request.files:
            return
        if not request.content:
            request.content = "I uploaded, just text me back and confirm that you got the file(s)."
        assert isinstance(request.content, str), "TODO: implement image support"
        request.content += "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            request.content += f"[Attachment: {file.name}]\n"
            self.ci_params.codebox.upload(file.name, file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    async def _ainput_handler(self, request: UserRequest) -> None:
        # TODO: variables as context to the agent
        # TODO: current files as context to the agent
        if not request.files:
            return
        if not request.content:
            request.content = "I uploaded, just text me back and confirm that you got the file(s)."
        assert isinstance(request.content, str), "TODO: implement image support"
        request.content += "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            request.content += f"[Attachment: {file.name}]\n"
            await self.ci_params.codebox.aupload(file.name, file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    def _output_handler_pre(self, response: Any) -> str:
        print("response(type)=", type(response))
        print("response=", response)
        if isinstance(response, str):
            output_str = response
        elif "output" in response:
            output_str = response["output"]
        else:
            output_str = "response=" + str(response)
        print("generate_response brain.invoke output_str=", output_str)
        return output_str

    def _output_handler_post(self, final_response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        for file in self.output_files:
            if str(file.name) in final_response:
                # rm ![Any](file.name) from the response
                final_response = re.sub(r"\n\n!\[.*\]\(.*\)", "", final_response)

        # if self.output_files and re.search(r"\n\[.*\]\(.*\)", final_response):
        #     try:
        #         final_response = remove_download_link(final_response, self.llm)
        #     except Exception as e:
        #         if self.verbose:
        #             print("Error while removing download links:", e)

        output_files = self.output_files
        code_log = self.code_log
        self.output_files = []
        self.code_log = []

        print("_output_handler self.brain.current_agent=", self.brain.current_agent)
        response = CodeInterpreterResponse(
            content=final_response, files=output_files, code_log=code_log, agent_name=self.brain.current_agent
        )
        return response

    def _output_handler(self, response: Any) -> CodeInterpreterResponse:
        """Embed images in the response"""
        final_response = self._output_handler_pre(response)
        response = self._output_handler_post(final_response)
        return response

    async def _aoutput_handler(self, final_response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        for file in self.output_files:
            if str(file.name) in final_response:
                # rm ![Any](file.name) from the response
                final_response = re.sub(r"\n\n!\[.*\]\(.*\)", "", final_response)

        if self.output_files and re.search(r"\n\[.*\]\(.*\)", final_response):
            try:
                final_response = await aremove_download_link(final_response, self.llm)
            except Exception as e:
                if self.verbose:
                    print("Error while removing download links:", e)

        output_files = self.output_files
        code_log = self.code_log
        self.output_files = []
        self.code_log = []

        response = CodeInterpreterResponse(content=final_response, files=output_files, code_log=code_log)
        return response

    def generate_response_sync(
        self,
        user_msg: str,
        files: list[File] = [],
    ) -> CodeInterpreterResponse:
        print("DEPRECATION WARNING: Use generate_response for sync generation.\n")
        return self.generate_response(
            user_msg=user_msg,
            files=files,
        )

    def generate_response(
        self,
        user_msg: str,
        files: list[File] = [],
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        user_request = UserRequest(content=user_msg, files=files)
        try:
            self._input_handler(user_request)
            print("user_request.content=", user_request.content)
            agent_scratchpad = ""
            input_message = {"input": user_request.content, "agent_scratchpad": agent_scratchpad}
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = self.brain.invoke(input=input_message)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            return self._output_handler(response)
        except Exception as e:
            traceback_str = "\n"
            if self.verbose:
                traceback.print_exc()
                traceback_str = traceback.print_list
            if settings.DETAILED_ERROR:
                return CodeInterpreterResponse(
                    content="Error in CodeInterpreterSession: " + f"{e.__class__.__name__}  - {e}" + traceback_str,
                    agent_name=self.brain.current_agent,
                )
            else:
                return CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session.",
                    agent_name=self.brain.current_agent,
                )

    async def agenerate_response(
        self,
        user_msg: str,
        files: list[File] = [],
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        user_request = UserRequest(content=user_msg, files=files)
        try:
            await self._ainput_handler(user_request)

            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = await self.brain.ainvoke(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======

            output_str = self._output_handler_pre(response)
            return await self._aoutput_handler(output_str)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                return CodeInterpreterResponse(
                    content="Error in CodeInterpreterSession(agenerate_response): " f"{e.__class__.__name__}  - {e}",
                    agent_name=self.brain.current_agent,
                )
            else:
                return CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session.",
                    agent_name=self.brain.current_agent,
                )

    def generate_response_stream(
        self,
        user_msg: str,
        files: list[File] = None,
    ) -> Iterator[str]:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            self._input_handler(user_request)
            agent_scratchpad = ""
            input_message = {"input": user_request.content, "agent_scratchpad": agent_scratchpad}
            print("llm stream start")
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response_stream = self.brain.stream(input=input_message)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("llm stream response(type)=", type(response_stream))

            full_output = ""
            for chunk in response_stream:
                if isinstance(chunk, dict) and "output" in chunk:
                    output = chunk["output"]
                else:
                    output = str(chunk)
                yield output
                full_output += output

            print("generate_response_stream brain.stream full_output=", full_output)
            self._aoutput_handler(full_output)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                yield f"Error in CodeInterpreterSession(generate_response_stream): {e.__class__.__name__} - {e}"
            else:
                yield "Sorry, something went wrong while generating your response. Please try again or restart the session."

    async def agenerate_response_stream(
        self,
        user_msg: str,
        files: list[File] = None,
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            await self._ainput_handler(user_request)

            print("llm astream start")
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = self.brain.astream(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("llm astream response(type)=", type(response))
            print("llm astream response=", response)

            full_output = ""
            async for chunk in response:
                yield chunk
                full_output += chunk["output"]

            print("agenerate_response_stream brain.astream full_output=", full_output)
            await self._aoutput_handler(full_output)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                yield CodeInterpreterResponse(
                    content="Error in CodeInterpreterSession(agenerate_response_stream): "
                    f"{e.__class__.__name__}  - {e}"
                )
            else:
                yield CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session."
                )

    def is_running(self) -> bool:
        return self.ci_params.codebox.status() == "running"

    async def ais_running(self) -> bool:
        return await self.ci_params.codebox.astatus() == "running"

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def stop(self) -> SessionStatus:
        return SessionStatus.from_codebox_status(self.ci_params.codebox.stop())

    async def astop(self) -> SessionStatus:
        return SessionStatus.from_codebox_status(await self.ci_params.codebox.astop())

    def __enter__(self) -> "CodeInterpreterSession":
        if self.is_local:
            self.start_local()
        else:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.stop()

    async def __aenter__(self) -> "CodeInterpreterSession":
        await self.astart()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.astop()
