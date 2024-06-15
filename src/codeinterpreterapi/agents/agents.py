from langchain.agents import AgentExecutor, BaseSingleActionAgent, ConversationalAgent, ConversationalChatAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from codeinterpreterapi.agents.structured_chat.agent_executor import load_structured_chat_agent_executor
from codeinterpreterapi.agents.tool_calling.agent_executor import load_tool_calling_agent_executor
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.tools.tools import CodeInterpreterTools


class CodeInterpreterAgent:
    @staticmethod
    def choose_agent_executor(ci_params: CodeInterpreterParams) -> AgentExecutor:
        is_structured_chat = True

        if is_structured_chat:
            return load_structured_chat_agent_executor(ci_params)
        else:
            # tool_calling
            return load_tool_calling_agent_executor(ci_params)

    @staticmethod
    def choose_single_chat_agent(ci_params: CodeInterpreterParams) -> BaseSingleActionAgent:
        # TODO: replace LCEL version
        llm = ci_params.llm
        tools = ci_params.tools
        is_ja = ci_params.is_ja
        system_message = settings.SYSTEM_MESSAGE if is_ja else settings.SYSTEM_MESSAGE_JA
        if isinstance(llm, ChatOpenAI) or isinstance(llm, AzureChatOpenAI):
            print("choose_agent OpenAIFunctionsAgent")
            agent = OpenAIFunctionsAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=system_message,
                extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
            )
        elif isinstance(llm, ChatAnthropic):
            print("choose_agent ConversationalChatAgent(ANTHROPIC)")
            agent = ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=str(system_message.content),
            )
        elif isinstance(llm, ChatGoogleGenerativeAI):
            print("choose_agent ChatGoogleGenerativeAI(gemini-pro)")
            agent = ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                system_message=str(system_message.content),
            )
        else:
            print("choose_agent ConversationalAgent(default)")
            agent = ConversationalAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                prefix=str(system_message.content),
            )

        return agent

    @staticmethod
    def create_single_chat_agent_executor(ci_params: CodeInterpreterParams) -> AgentExecutor:
        # agent
        agent = CodeInterpreterAgent.choose_single_chat_agent(ci_params)

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


def test():
    sample = "ツールのpythonで円周率を表示するプログラムを実行してください。"
    # sample = "lsコマンドを実行してください。"
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    ci_params.tools = []
    ci_params.tools = CodeInterpreterTools(ci_params).get_all_tools()

    agent = CodeInterpreterAgent.choose_agent_executor(ci_params=ci_params)
    # agent = CodeInterpreterAgent.choose_single_chat_agent(ci_params=ci_params)
    # agent = CodeInterpreterAgent.create_agent_and_executor_experimental(ci_params=ci_params)
    result = agent.invoke({"input": sample})
    print("result=", result)


if __name__ == "__main__":
    test()
