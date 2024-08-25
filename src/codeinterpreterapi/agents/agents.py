import os
from typing import List

import yaml
from gui_agent_loop_core.schema.agent.schema import AgentDefinition, AgentType
from langchain.agents import AgentExecutor, BaseSingleActionAgent, ConversationalAgent, ConversationalChatAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import ValidationError

from codeinterpreterapi.agents.structured_chat.agent_executor import load_structured_chat_agent_executor
from codeinterpreterapi.agents.tool_calling.agent_executor import load_tool_calling_agent_executor
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt
from codeinterpreterapi.tools.tools import CodeInterpreterTools


class CodeInterpreterAgent:
    @staticmethod
    def choose_agent_executors(ci_params: CodeInterpreterParams) -> List[AgentExecutor]:
        CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
        with open(os.path.join(CONFIG_DIR, "agents_config.yaml"), "r", encoding="utf8") as f:
            agents_config = yaml.safe_load(f)

        agent_executors = []
        for agent in agents_config["agents"]:
            agent_name = agent["name"]
            config_path = agent["config_path"]

            with open(os.path.join(CONFIG_DIR, config_path), "r", encoding="utf8") as f:
                config = yaml.safe_load(f)

            try:
                print(f"choose_agent_executors Agent: {agent_name}")
                # print(f"config: {config}")
                agent_def = AgentDefinition(**config["agent_definition"])
                agent_def.build_prompt()
                print("agent_def=", agent_def)
                print("---")
                agent_executor = CodeInterpreterAgent.choose_agent_executor(ci_params, agent_def)
                agent_executors.append(agent_executor)
                agent_def.agent_executor = agent_executor
                ci_params.agent_def_list.append(agent_def)
            except ValidationError as e:
                print(f"設定ファイルの検証に失敗しました（Agent: {agent_name}）: {e}")

        return agent_executors

    @staticmethod
    def choose_agent_executor(ci_params: CodeInterpreterParams, agent_def: AgentDefinition = None) -> AgentExecutor:
        if agent_def.agent_type == AgentType.STRUCTURED_CHAT:
            return load_structured_chat_agent_executor(ci_params, agent_def)
        else:
            # tool_calling
            return load_tool_calling_agent_executor(ci_params, agent_def)

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
            # callbacks=ci_params.callbacks,
        )
        print("agent_executor.input_keys", agent_executor.input_keys)
        print("agent_executor.output_keys", agent_executor.output_keys)
        return agent_executor


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    ci_params.tools = CodeInterpreterTools(ci_params).get_all_tools()
    agent_executors = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    result = agent_executors[0].invoke({"input": TestPrompt.svg_input_str})
    print("result=", result)


if __name__ == "__main__":
    test()
