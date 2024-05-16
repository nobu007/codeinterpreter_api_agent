import pprint

from langchain.agents import AgentExecutor, BaseSingleActionAgent, ConversationalAgent, ConversationalChatAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory.buffer import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from codeinterpreterapi.agents.plan_and_execute.agent_executor import load_agent_executor
from codeinterpreterapi.config import settings


class CodeInterpreterAgent:
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
    def create_agent_and_executor(llm, tools, verbose, chat_memory, callbacks) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(llm, tools)
        print("create_agent_and_executor agent=", str(type(agent)))
        # pprint.pprint(agent)

        # agent_executor
        agent_executor = load_agent_executor(
            agent=agent,
            max_iterations=settings.MAX_ITERATIONS,
            tools=tools,
            verbose=verbose,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=chat_memory,
            ),
            callbacks=callbacks,
        )
        print("create_agent_and_executor agent_executor tools:")
        for tool in agent_executor.tools:
            pprint.pprint(tool)

        return agent_executor

    @staticmethod
    def create_agent_and_executor_experimental(llm, tools, verbose, is_ja) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(llm, tools, is_ja)
        print("create_agent_and_executor agent=", str(type(agent)))

        # agent_executor
        agent_executor = load_agent_executor(llm, tools, verbose=verbose, is_ja=is_ja)

        return agent_executor
