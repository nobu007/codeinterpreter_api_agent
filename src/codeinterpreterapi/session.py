import base64
import re
import subprocess
import tempfile
import traceback
from io import BytesIO
from types import TracebackType
from typing import Any, Optional, Type
from uuid import UUID, uuid4

from codeboxapi import CodeBox  # type: ignore
from codeboxapi.schema import CodeBoxOutput  # type: ignore
from gui_agent_loop_core.schema.schema import GuiAgentInterpreterChatResponseStr
from langchain.agents import AgentExecutor
from langchain.callbacks.base import Callbacks
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.chains import (
    aget_file_modifications,
    aremove_download_link,
    get_file_modifications,
    remove_download_link,
)
from codeinterpreterapi.chat_history import CodeBoxChatMessageHistory
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import CodeInterpreterLlm
from codeinterpreterapi.schema import CodeInterpreterResponse, File, SessionStatus, UserRequest
from codeinterpreterapi.thoughts.thoughts import CodeInterpreterToT

from .planners.planners import CodeInterpreterPlanner
from .supervisors.supervisors import CodeInterpreterSupervisor
from .tools.tools import CodeInterpreterTools


def _handle_deprecated_kwargs(kwargs: dict) -> None:
    settings.MODEL = kwargs.get("model", settings.MODEL)
    settings.MAX_RETRY = kwargs.get("max_retry", settings.MAX_RETRY)
    settings.TEMPERATURE = kwargs.get("temperature", settings.TEMPERATURE)
    settings.OPENAI_API_KEY = kwargs.get("openai_api_key", settings.OPENAI_API_KEY)
    settings.SYSTEM_MESSAGE = kwargs.get("system_message", settings.SYSTEM_MESSAGE)
    settings.MAX_ITERATIONS = kwargs.get("max_iterations", settings.MAX_ITERATIONS)


class CodeInterpreterSession:
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        additional_tools: list[BaseTool] = [],
        callbacks: Callbacks = None,
        is_local: bool = True,
        is_ja: bool = True,
        **kwargs: Any,
    ) -> None:
        _handle_deprecated_kwargs(kwargs)
        self.is_local = is_local
        self.is_ja = is_ja
        self.codebox = CodeBox(requirements=settings.CUSTOM_PACKAGES)
        self.verbose = kwargs.get("verbose", settings.DEBUG)
        run_handler_func = self._run_handler
        arun_handler_func = self._arun_handler
        if self.is_local:
            run_handler_func = self._run_handler_local
            arun_handler_func = self._arun_handler_local
        self.llm: BaseLanguageModel = llm or CodeInterpreterLlm.get_llm()
        self.tools: list[BaseTool] = CodeInterpreterTools(
            additional_tools, run_handler_func, arun_handler_func, self.llm
        ).get_all_tools()
        self.log("self.llm=" + str(self.llm))

        self.callbacks = callbacks
        self.agent_executor: Optional[Runnable] = None
        self.llm_planner: Optional[Runnable] = None
        self.supervisor: Optional[AgentExecutor] = None
        self.thought: Optional[Runnable] = None
        self.input_files: list[File] = []
        self.output_files: list[File] = []
        self.code_log: list[tuple[str, str]] = []

    @classmethod
    def from_id(cls, session_id: UUID, **kwargs: Any) -> "CodeInterpreterSession":
        session = cls(**kwargs)
        session.codebox = CodeBox.from_id(session_id)
        session.agent_executor = CodeInterpreterAgent.create_agent_and_executor()
        return session

    @property
    def session_id(self) -> Optional[UUID]:
        return self.codebox.session_id

    def initialize(self):
        self.initialize_agent_executor()
        self.initialize_llm_planner()
        self.initialize_supervisor()
        self.initialize_thought()

    def initialize_agent_executor(self):
        is_experimental = False
        if is_experimental:
            self.agent_executor = CodeInterpreterAgent.create_agent_and_executor_experimental(
                llm=self.llm,
                tools=self.tools,
                verbose=self.verbose,
                is_ja=self.is_ja,
            )
        else:
            self.agent_executor = CodeInterpreterAgent.create_agent_executor_lcel(
                llm=self.llm,
                tools=self.tools,
                verbose=self.verbose,
                is_ja=self.is_ja,
                chat_memory=self._history_backend(),
                callbacks=self.callbacks,
            )

    def initialize_llm_planner(self):
        self.llm_planner = CodeInterpreterPlanner.choose_planner(llm=self.llm, tools=self.tools, is_ja=self.is_ja)

    def initialize_supervisor(self):
        self.supervisor = CodeInterpreterSupervisor.choose_supervisor(
            planner=self.llm_planner,
            executor=self.agent_executor,
            tools=self.tools,
            verbose=self.verbose,
        )

    def initialize_thought(self):
        self.thought = CodeInterpreterToT.get_runnable_tot_chain(llm=self.llm, is_ja=self.is_ja, is_simple=False)

    def start(self) -> SessionStatus:
        print("start")
        status = SessionStatus.from_codebox_status(self.codebox.start())
        self.initialize()
        self.codebox.run(
            f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
        )
        return status

    async def astart(self) -> SessionStatus:
        print("astart")
        status = SessionStatus.from_codebox_status(await self.codebox.astart())
        self.initialize()
        await self.codebox.arun(
            f"!pip install -q {' '.join(settings.CUSTOM_PACKAGES)}",
        )
        return status

    def start_local(self) -> SessionStatus:
        print("start_local")
        self.initialize()
        status = SessionStatus(status="started")
        return status

    async def astart_local(self) -> SessionStatus:
        print("astart_local")
        status = self.start_local()
        self.initialize()
        return status

    def _history_backend(self) -> BaseChatMessageHistory:
        return (
            CodeBoxChatMessageHistory(codebox=self.codebox)
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

    def show_code(self, code: str) -> None:
        if self.verbose:
            print(code)

    async def ashow_code(self, code: str) -> None:
        """Callback function to show code to the user."""
        if self.verbose:
            print(code)

    def _get_handler_local_command(self, code: str):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

            command = f"cd src/codeinterpreterapi/invoke_tasks && invoke -c python run-code-file '{temp_file_path}'"
            return command

    def _run_handler_local(self, code: str):
        print("_run_handler_local code=", code)
        command = self._get_handler_local_command(code)
        try:
            output_content = subprocess.check_output(command, shell=True, universal_newlines=True)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            return None

    async def _arun_handler_local(self, code: str):
        print("_arun_handler_local code=", code)
        command = self._get_handler_local_command(code)
        try:
            output_content = await subprocess.check_output(command, shell=True, universal_newlines=True)
            self.code_log.append((code, output_content))
            return output_content
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            return None

    def _run_handler(self, code: str) -> str:
        """Run code in container and send the output to the user"""
        self.show_code(code)
        output: CodeBoxOutput = self.codebox.run(code)
        self.code_log.append((code, output.content))

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'",
                    output.content,
                ):
                    self.codebox.install(package.group(1))
                    return f"{package.group(1)} was missing but " "got installed now. Please try again."
            else:
                # TODO: pre-analyze error to optimize next code generation
                pass
            if self.verbose:
                print("Error:", output.content)

        elif modifications := get_file_modifications(code, self.llm):
            for filename in modifications:
                if filename in [file.name for file in self.input_files]:
                    continue
                fileb = self.codebox.download(filename)
                if not fileb.content:
                    continue
                file_buffer = BytesIO(fileb.content)
                file_buffer.name = filename
                self.output_files.append(File(name=filename, content=file_buffer.read()))

        return output.content

    async def _arun_handler(self, code: str) -> str:
        """Run code in container and send the output to the user"""
        await self.ashow_code(code)
        output: CodeBoxOutput = await self.codebox.arun(code)
        self.code_log.append((code, output.content))

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'",
                    output.content,
                ):
                    await self.codebox.ainstall(package.group(1))
                    return f"{package.group(1)} was missing but " "got installed now. Please try again."
            else:
                # TODO: pre-analyze error to optimize next code generation
                pass
            if self.verbose:
                print("Error:", output.content)

        elif modifications := await aget_file_modifications(code, self.llm):
            for filename in modifications:
                if filename in [file.name for file in self.input_files]:
                    continue
                fileb = await self.codebox.adownload(filename)
                if not fileb.content:
                    continue
                file_buffer = BytesIO(fileb.content)
                file_buffer.name = filename
                self.output_files.append(File(name=filename, content=file_buffer.read()))

        return output.content

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
            self.codebox.upload(file.name, file.content)
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
            await self.codebox.aupload(file.name, file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    def _output_handler(self, final_response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        for file in self.output_files:
            if str(file.name) in final_response:
                # rm ![Any](file.name) from the response
                final_response = re.sub(r"\n\n!\[.*\]\(.*\)", "", final_response)

        if self.output_files and re.search(r"\n\[.*\]\(.*\)", final_response):
            try:
                final_response = remove_download_link(final_response, self.llm)
            except Exception as e:
                if self.verbose:
                    print("Error while removing download links:", e)

        output_files = self.output_files
        code_log = self.code_log
        self.output_files = []
        self.code_log = []

        response = CodeInterpreterResponse(content=final_response, files=output_files, code_log=code_log)
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
            assert self.agent_executor, "Session not initialized."
            print("user_request.content=", user_request.content)
            agent_scratchpad = ""
            input_message = {"input": user_request.content, "agent_scratchpad": agent_scratchpad}

            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            # response = self.agent_executor.invoke(input=input_message)
            # response = self.llm_planner.invoke(input=input_message)
            # response = self.supervisor.invoke(input=input_message)
            response = self.thought.invoke(input=input_message)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("response(type)=", type(response))
            print("response=", response)

            output = response["output"]
            print("generate_response agent_executor.invoke output=", output)
            return self._output_handler(output)
            # return output
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                return CodeInterpreterResponse(
                    content="Error in CodeInterpreterSession: " f"{e.__class__.__name__}  - {e}"
                )
            else:
                return CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session."
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
            assert self.agent_executor, "Session not initialized."

            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = await self.agent_executor.ainvoke(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======

            output = response["output"]
            print("agenerate_response agent_executor.ainvoke output=", output)
            return await self._aoutput_handler(output)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                return CodeInterpreterResponse(
                    content="Error in CodeInterpreterSession(agenerate_response): " f"{e.__class__.__name__}  - {e}"
                )
            else:
                return CodeInterpreterResponse(
                    content="Sorry, something went while generating your response."
                    "Please try again or restart the session."
                )

    def generate_response_stream(
        self,
        user_msg: str,
        files: list[File] = None,
    ) -> GuiAgentInterpreterChatResponseStr:
        """Generate a Code Interpreter response based on the user's input."""
        if files is None:
            files = []
        user_request = UserRequest(content=user_msg, files=files)
        try:
            self._input_handler(user_request)
            assert self.agent_executor, "Session not initialized."
            print("llm stream start")
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = self.agent_executor.stream(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("llm stream response(type)=", type(response))
            print("llm stream response=", response)

            full_output = ""
            for chunk in response:
                yield chunk
                full_output += chunk["output"]

            print("generate_response_stream agent_executor.stream full_output=", full_output)
            self._aoutput_handler(full_output)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            if settings.DETAILED_ERROR:
                yield "Error in CodeInterpreterSession(generate_response_stream): " f"{e.__class__.__name__}  - {e}"
            else:
                yield (
                    "Sorry, something went while generating your response." + "Please try again or restart the session."
                )

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
            assert self.agent_executor, "Session not initialized."

            print("llm astream start")
            # ======= ↓↓↓↓ LLM invoke ↓↓↓↓ #=======
            response = self.agent_executor.astream(input=user_request.content)
            # ======= ↑↑↑↑ LLM invoke ↑↑↑↑ #=======
            print("llm astream response(type)=", type(response))
            print("llm astream response=", response)

            full_output = ""
            async for chunk in response:
                yield chunk
                full_output += chunk["output"]

            print("agent_executor.astream full_output=", full_output)
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
        return self.codebox.status() == "running"

    async def ais_running(self) -> bool:
        return await self.codebox.astatus() == "running"

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def stop(self) -> SessionStatus:
        return SessionStatus.from_codebox_status(self.codebox.stop())

    async def astop(self) -> SessionStatus:
        return SessionStatus.from_codebox_status(await self.codebox.astop())

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
