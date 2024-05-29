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
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings


class CodeInterpreterAgent:
    @staticmethod
    def create_agent_executor_lcel(ci_params: CodeInterpreterParams) -> Runnable:
        # prompt
        prompt = hub.pull("hwchase17/openai-functions-agent")

        # agent
        agent = create_tool_calling_agent(ci_params.llm, ci_params.tools, prompt)

        # agent_executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=ci_params.tools,
            verbose=ci_params.verbose,
            # memory=ConversationBufferMemory(
            #     memory_key="chat_history",
            #     return_messages=True,
            #     chat_memory=chat_memory,
            # ),
            callbacks=ci_params.callbacks,
        )
        print("agent_executor.input_keys", agent_executor.input_keys)
        print("agent_executor.output_keys", agent_executor.output_keys)
        return agent_executor

    @staticmethod
    def choose_single_chat_agent(ci_params: CodeInterpreterParams) -> BaseSingleActionAgent:
        llm = ci_params.llm
        tools = ci_params.tools
        is_ja = ci_params.is_ja
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
    def create_agent_and_executor(ci_params: CodeInterpreterParams) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(ci_params)
        print("create_agent_and_executor agent=", str(type(agent)))
        return agent

    @staticmethod
    def create_agent_and_executor_experimental(ci_params: CodeInterpreterParams) -> AgentExecutor:
        # agent_executor
        agent_executor = load_agent_executor(ci_params)
        print("create_agent_and_executor_experimental")

        return agent_executor
