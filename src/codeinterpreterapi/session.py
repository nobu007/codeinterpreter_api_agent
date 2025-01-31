import re
import traceback
from types import TracebackType
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Type
from uuid import UUID
import time


from codeboxapi import CodeBox  # type: ignore
from codeboxapi.schema import CodeBoxStatus  # type: ignore
from gui_agent_loop_core.schema.message.schema import BaseMessageContent
from langchain.callbacks.base import Callbacks
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from codeinterpreterapi.brain.brain import CodeInterpreterBrain
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.callbacks.markdown.callbacks import MarkdownFileCallbackHandler
from codeinterpreterapi.chains import aremove_download_link
from codeinterpreterapi.chat_history import CodeBoxChatMessageHistory
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import CodeInterpreterLlm
from codeinterpreterapi.schema import CodeInterpreterResponse, File, SessionStatus, UserRequest
from codeinterpreterapi.utils.multi_converter import MultiConverter


def _handle_deprecated_kwargs(kwargs: dict) -> None:
    settings.MODEL = kwargs.get("model", settings.MODEL)
    settings.MAX_RETRY = kwargs.get("max_retry", settings.MAX_RETRY)
    settings.TEMPERATURE = kwargs.get("temperature", settings.TEMPERATURE)
    settings.OPENAI_API_KEY = kwargs.get("openai_api_key", settings.OPENAI_API_KEY)
    settings.SYSTEM_MESSAGE = kwargs.get("system_message", settings.SYSTEM_MESSAGE)
    settings.MAX_ITERATIONS = kwargs.get("max_iterations", settings.MAX_ITERATIONS)


class AgentCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self):
        self.messages = []  # チェーン内の応答を格納
        self.complete = False  # チェーン終了フラグ

    def stream_responses(self):
        """
        ポーリング的に応答をストリームするジェネレータ関数
        """
        while not self.complete or len(self.messages) > 0:
            if self.messages:
                yield self.messages.pop(0)  # メッセージを逐次取得
            else:
                time.sleep(0.5)  # メッセージがない場合の待機

    ### on_chain callbacks ###
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        print("AgentCallbackHandler on_chain_start run_id=", run_id)
        self.complete = False

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_chain_end run_id=", run_id, ", type(outputs)=", type(outputs))
        self.messages.append(outputs)  # 最終応答を格納
        self.complete = True  # 終了フラグを設定

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_chain_error")
        self.messages.append(str(error))  # エラーを格納
        self.complete = True  # 終了フラグを設定

    ### on_chat callbacks ###
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
        """Run when a chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        print("AgentCallbackHandler on_chat_model_start")

    ### on_agent callbacks ###

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_agent_action")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_agent_finish")

    ### on_tool callbacks ###
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool starts running.

        Args:
            serialized (Dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            inputs (Optional[Dict[str, Any]]): The inputs.
            kwargs (Any): Additional keyword arguments.
        """
        print("AgentCallbackHandler on_tool_start")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_tool_end")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments."""
        print("AgentCallbackHandler on_tool_error")


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
        self.agent_callback_handler = AgentCallbackHandler()
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
            configurable=configurable,
            # callbacks=[],
            callbacks=[self.agent_callback_handler, MarkdownFileCallbackHandler("langchain_log.md")],
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
        self.output_code_log_list: list[tuple[str, str]] = []

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
        codebox_status = CodeBoxStatus(status="unknown")
        if self.ci_params.codebox:
            codebox_status = self.ci_params.codebox.start()
            self.ci_params.codebox.run(
                f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
            )
        self.brain.initialize()
        return SessionStatus.from_codebox_status(codebox_status)

    async def astart(self) -> SessionStatus:
        print("astart")
        codebox_status = CodeBoxStatus(status="unknown")
        if self.ci_params.codebox:
            codebox_status = self.ci_params.codebox.astart()
            self.ci_params.codebox.arun(
                f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
            )
        self.brain.initialize()
        return SessionStatus.from_codebox_status(codebox_status)

    def start_local(self) -> SessionStatus:
        # TODO: delete it and use start()
        print("start_local")
        self.brain.initialize()
        status = SessionStatus(status="started")
        return status

    async def astart_local(self) -> SessionStatus:
        # TODO: delete it and use astart()
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

    def _input_message_prepare(self, request: UserRequest) -> None:
        # set return input_message
        if isinstance(request.content, str):
            input_message = {"input": request.content, "agent_scratchpad": ""}
        else:
            input_message = request.content
        return input_message

    def _input_handler_common(self, request: BaseMessageContent, add_content_str: str) -> None:
        """Callback function to handle user input."""
        # TODO: variables as context to the agent
        # TODO: current files as context to the agent
        if not request.files:
            return self._input_message_prepare(request)
        if not request.content:
            add_content_str = "I uploaded, just text me back and confirm that you got the file(s).\n" + add_content_str
        assert isinstance(request.content, BaseMessageContent)  # "TODO: implement image support"

        # set request.content
        if isinstance(request.content, str):
            request.content += add_content_str
        elif isinstance(request.content, list):
            last_content = request.content[-1]
            if isinstance(last_content, str):
                last_content += add_content_str
            else:
                pass
                # TODO impl it.
                # last_content["input"] += add_content_str
        else:
            # Dict
            pass
            # TODO impl it.
            # request.content["input"] += add_content_str
        return self._input_message_prepare(request)

    def _input_handler(self, request: UserRequest) -> None:
        """Callback function to handle user input."""
        add_content_str = "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            add_content_str += f"[Attachment: {file.name}]\n"
            self.ci_params.codebox.upload(file.name, file.content)
        add_content_str += "**File(s) are now available in the cwd. **\n"
        return self._input_handler_common(request, add_content_str)

    async def _ainput_handler(self, request: UserRequest) -> None:
        """Callback function to handle user input."""
        add_content_str = "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            add_content_str += f"[Attachment: {file.name}]\n"
            await self.ci_params.codebox.aupload(file.name, file.content)
        add_content_str += "**File(s) are now available in the cwd. **\n"
        return self._input_handler_common(request, add_content_str)

    def _output_handler_pre(self, response: Any) -> str:
        output_str = MultiConverter.to_str(response)

        # TODO: MultiConverterに共通化
        if isinstance(response, dict):
            output_str = ""
            code_log_item = {}
            if "output" in response:
                output_str = response["output"]
            if "tool" in response:
                code_log_item["tool"] = str(response["tool"])
            if "tool_input" in response:
                code_log_item["tool_input"] = str(response["tool_input"])
            if "log" in response:
                code_log_item["log"] = str(response["log"])
            self.output_code_log_list = code_log_item
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
        code_log = self.output_code_log_list
        final_code = ""
        final_log = ""
        if len(final_code) > 0:
            final_code = code_log[-1][0]
            final_log = code_log[-1][0]
        self.output_files = []
        self.output_code_log_list = []

        response = CodeInterpreterResponse(
            content=final_response,
            files=output_files,
            code_log=code_log,
            agent_name=self.brain.current_agent,
            code=final_code,
            log=final_log,
        )
        return response

    def _output_handler(self, response: Any) -> CodeInterpreterResponse:
        """Embed images in the response"""
        print("XXXX _output_handler in response=", type(response))
        final_response = self._output_handler_pre(response)
        print("XXXX _output_handler step1 ")
        response = self._output_handler_post(final_response)
        print("XXXX _output_handler out ")
        return response

    async def _aoutput_handler(self, response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        print("XXXX _aoutput_handler in response=", type(response))
        final_response = self._output_handler_pre(response)
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
        code_log = self.output_code_log_list
        self.output_files = []
        self.output_code_log_list = []

        response = CodeInterpreterResponse(content=final_response, files=output_files, code_log=code_log)
        return response

    def generate_response_sync(
        self,
        user_msg: BaseMessageContent,
        files: list[File] = [],
    ) -> CodeInterpreterResponse:
        print("DEPRECATION WARNING: Use generate_response for sync generation.\n")
        return self.generate_response(
            user_msg=user_msg,
            files=files,
        )

    def generate_response(
        self,
        user_msg: BaseMessageContent,
        files: list[File] = None,
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            input_message = self._input_handler(user_request)
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
        user_msg: BaseMessageContent,
        files: list[File] = None,
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            input_message = await self._ainput_handler(user_request)
            agent_scratchpad = ""
            input_message = {"input": user_request.content, "agent_scratchpad": agent_scratchpad}
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = await self.brain.ainvoke(input=input_message)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            return await self._aoutput_handler(response)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                return CodeInterpreterResponse(
                    content=f"Error in CodeInterpreterSession(agenerate_response): {e.__class__.__name__}  - {e}",
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
        user_msg: BaseMessageContent,
        files: list[File] = None,
    ) -> Iterator[CodeInterpreterResponse]:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            input_message = self._input_handler(user_request)
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response_stream = self.brain.stream(input=input_message)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("generate_response_stream type(response_stream)=", type(response_stream))

            # wait for the stream_responses from agent
            for stream_response in self.agent_callback_handler.stream_responses():
                ci_response = self._output_handler(stream_response)
                print("generate_response_stream ci_response(agent)=", type(stream_response))
                yield ci_response

            # wait for the final response
            for chunk in response_stream:
                if isinstance(chunk, dict) and "output" in chunk:
                    output = chunk["output"]
                else:
                    output = str(chunk)

                print("generate_response_stream ci_response(output)=", type(chunk))
                ci_response = self._output_handler(output)
                yield ci_response

        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                yield CodeInterpreterResponse(
                    content=f"Error in CodeInterpreterSession(generate_response_stream): {e.__class__.__name__} - {e}"
                )
            else:
                yield CodeInterpreterResponse(
                    content="Sorry, something went wrong while generating your response. Please try again or restart the session."
                )
        finally:
            yield CodeInterpreterResponse(content="", end=True, agent_name=self.brain.current_agent)

    async def agenerate_response_stream(
        self,
        user_msg: BaseMessageContent,
        files: list[File] = None,
    ) -> AsyncGenerator[CodeInterpreterResponse, None]:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            await self._ainput_handler(user_request)

            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = self.brain.astream(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======

            async for chunk in response:
                print("agenerate_response_stream brain.astream chunk=", chunk)
                ci_response: CodeInterpreterResponse = await self._aoutput_handler(chunk)
                yield ci_response
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                yield CodeInterpreterResponse(
                    content=f"Error in CodeInterpreterSession(agenerate_response_stream): {e.__class__.__name__}  - {e}"
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
        codebox_status = CodeBoxStatus(status="unknown")
        if self.ci_params.codebox:
            codebox_status = self.ci_params.codebox.stop()
        return SessionStatus.from_codebox_status(codebox_status)

    async def astop(self) -> SessionStatus:
        codebox_status = CodeBoxStatus(status="unknown")
        if self.ci_params.codebox:
            codebox_status = await self.ci_params.codebox.astop()
        return SessionStatus.from_codebox_status(codebox_status)

    def __enter__(self) -> "CodeInterpreterSession":
        if self.ci_params.is_local:
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
        await self.astop()
