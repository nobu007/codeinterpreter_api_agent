import pprint

from langchain.agents import (
    AgentExecutor,
    BaseSingleActionAgent,
    ConversationalAgent,
    ConversationalChatAgent,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models.base import BaseChatModel
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from codeinterpreterapi.config import settings


class CodeInterpreterAgent:

    @staticmethod
    def choose_agent(
        llm,
        tools,
    ) -> BaseSingleActionAgent:
        if isinstance(llm, ChatOpenAI) or isinstance(llm, AzureChatOpenAI):
            print("choose_agent OpenAIFunctionsAgent")
            return OpenAIFunctionsAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=settings.SYSTEM_MESSAGE,
                extra_prompt_messages=[
                    MessagesPlaceholder(variable_name="chat_history")
                ],
            )
        elif isinstance(llm, BaseChatModel):
            print("choose_agent ConversationalChatAgent(ANTHROPIC)")
            return ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=settings.SYSTEM_MESSAGE.content.__str__(),
            )
        else:
            print("choose_agent ConversationalAgent(default)")
            return ConversationalAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                prefix=settings.SYSTEM_MESSAGE.content.__str__(),
            )

    @staticmethod
    def create_agent_and_executor(
        llm, tools, verbose, chat_memory, callbacks
    ) -> AgentExecutor:

        # agent
        agent = CodeInterpreterAgent.choose_agent(llm, tools)
        print("create_agent_and_executor agent=", str(type(agent)))
        # pprint.pprint(agent)

        # agent_executor
        agent_executor = AgentExecutor.from_agent_and_tools(
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
