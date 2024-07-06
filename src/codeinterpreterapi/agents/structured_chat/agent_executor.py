# agent_executor.py
# https://github.com/langchain-ai/langchain/blob/3ee07473821906a29d944866a2ededb41148f234/libs/experimental/langchain_experimental/plan_and_execute/executors/agent_executor.py

from gui_agent_loop_core.schema.agent.schema import AgentDefinition
from langchain.agents.agent import AgentExecutor

from codeinterpreterapi.agents.structured_chat.agent import create_structured_chat_agent
from codeinterpreterapi.agents.structured_chat.prompts import create_structured_chat_agent_prompt
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm


def load_structured_chat_agent_executor(
    ci_params: CodeInterpreterParams, agent_def: AgentDefinition = None
) -> AgentExecutor:
    """
    Load an agent executor(general purpose).
    """
    prompt = create_structured_chat_agent_prompt(ci_params.is_ja)
    if agent_def.agent_role is not None:
        prompt = prompt.partial(agent_role=agent_def.agent_role)
    input_variables = prompt.input_variables
    if ci_params.verbose_prompt:
        print("load_structured_chat_agent_executor prompt.input_variables=", input_variables)
        print("load_structured_chat_agent_executor prompt=", prompt.messages)
    agent = create_structured_chat_agent(
        llm=ci_params.llm_tools,
        tools=ci_params.tools,
        # output_parser=output_parser,
        prompt=prompt,
        runnable_config=ci_params.runnable_config,
        # stop_sequence=["Observation:", "最終回答", "Final Answer"],
    )
    agent_def.agent = agent

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ci_params.tools, verbose=ci_params.verbose)
    agent_def.agent_executor = agent_executor
    return agent_executor


def test():
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    agent_executor = load_structured_chat_agent_executor(ci_params)
    test_input = "pythonで円周率を表示するプログラムを実行してください。"
    agent_executor_output = agent_executor.invoke({"input": test_input})
    print("agent_executor_output=", agent_executor_output)


if __name__ == "__main__":
    test()
