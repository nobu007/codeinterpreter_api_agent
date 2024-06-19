# agent_executor.py
# https://github.com/langchain-ai/langchain/blob/3ee07473821906a29d944866a2ededb41148f234/libs/experimental/langchain_experimental/plan_and_execute/executors/agent_executor.py

from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from codeinterpreterapi.agents.tool_calling.agent import create_tool_calling_agent
from codeinterpreterapi.agents.tool_calling.prompts import create_tool_calling_agent_prompt
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm


def load_tool_calling_agent_executor(
    ci_params: CodeInterpreterParams, prompt: ChatPromptTemplate = None
) -> AgentExecutor:
    """
    Load an agent executor(general purpose).
    """
    prompt = None
    if prompt is None:
        prompt = create_tool_calling_agent_prompt(ci_params.is_ja)
    input_variables = prompt.input_variables
    print("load_tool_calling_agent_executor prompt.input_variables=", input_variables)
    agent = create_tool_calling_agent(
        llm=ci_params.llm,
        tools=ci_params.tools,
        # output_parser=output_parser,
        prompt=prompt,
        runnable_config=ci_params.runnable_config,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ci_params.tools, verbose=ci_params.verbose)
    return agent_executor


def test():
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    agent_executor = load_tool_calling_agent_executor(ci_params)
    test_input = "pythonで円周率を表示するプログラムを実行してください。"
    agent_executor_output = agent_executor.invoke({"input": test_input})
    print("agent_executor_output=", agent_executor_output)


if __name__ == "__main__":
    test()
