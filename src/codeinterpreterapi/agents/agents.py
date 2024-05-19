import getpass
import os
import platform

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    BaseSingleActionAgent,
    ConversationalAgent,
    ConversationalChatAgent,
    create_tool_calling_agent,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from codeinterpreterapi.agents.plan_and_execute.agent_executor import load_agent_executor
from codeinterpreterapi.config import settings


class CodeInterpreterAgent:
    @staticmethod
    def create_agent_executor_lcel(llm, tools, verbose=False, chat_memory=None, callbacks=None, is_ja=True) -> Runnable:
        # agent
        prompt = hub.pull("hwchase17/openai-functions-agent")
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        # llm_with_tools = llm.bind_tools(tools)
        # llm_with_tools_and_info = llm_with_tools.bind({"agent_scratchpad": info})

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
        )
        print("agent_executor.input_keys", agent_executor.input_keys)
        print("agent_executor.output_keys", agent_executor.output_keys)
        return agent_executor

    @staticmethod
    def choose_single_chat_agent(
        llm,
        tools,
        is_ja,
    ) -> BaseSingleActionAgent:
        system_message = settings.SYSTEM_MESSAGE if is_ja else settings.SYSTEM_MESSAGE_JA
        if isinstance(llm, ChatOpenAI) or isinstance(llm, AzureChatOpenAI):
            print("choose_agent OpenAIFunctionsAgent")
            return OpenAIFunctionsAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=system_message,
                extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
            )
        elif isinstance(llm, ChatAnthropic):
            print("choose_agent ConversationalChatAgent(ANTHROPIC)")
            return ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=str(system_message.content),
            )
        elif isinstance(llm, ChatGoogleGenerativeAI):
            print("choose_agent ChatGoogleGenerativeAI(gemini-pro)")
            return ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=str(system_message.content),
            )
        else:
            print("choose_agent ConversationalAgent(default)")
            return ConversationalAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                prefix=str(system_message.content),
            )

    @staticmethod
    def create_agent_and_executor(llm, tools, verbose, chat_memory, callbacks, is_ja=True) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(llm, tools, is_ja=is_ja)
        print("create_agent_and_executor agent=", str(type(agent)))
        return agent

    @staticmethod
    def create_agent_and_executor_experimental(llm, tools, verbose, is_ja) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(llm, tools, is_ja)
        print("create_agent_and_executor agent=", str(type(agent)))

        # agent_executor
        agent_executor = load_agent_executor(llm, tools, verbose=verbose, is_ja=is_ja)

        return agent_executor
