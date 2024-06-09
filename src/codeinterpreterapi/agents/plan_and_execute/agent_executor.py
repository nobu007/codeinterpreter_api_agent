from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import create_structured_chat_agent

from codeinterpreterapi.agents.plan_and_execute.prompts import create_structured_chat_agent_prompt
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm


def load_agent_executor(ci_params: CodeInterpreterParams) -> AgentExecutor:
    """
    Load an agent executor.

    Args:
        llm: BaseLanguageModel
        tools: List[BaseTool]
        verbose: bool. Defaults to False.
        include_task_in_prompt: bool. Defaults to False.

    Returns:
        ChainExecutor
    """
    prompt = create_structured_chat_agent_prompt(ci_params.is_ja)
    input_variables = prompt.input_variables
    print("load_agent_executor prompt.input_variables=", input_variables)
    agent = create_structured_chat_agent(
        llm=ci_params.llm,
        tools=ci_params.tools,
        # callback_manager=callback_manager,
        # output_parser=output_parser,
        # prefix=tools_prefix,
        # suffix=suffix,
        prompt=prompt,
        # format_instructions=format_instructions,
        # memory_prompts = memory_prompts,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=ci_params.tools, verbose=ci_params.verbose)
    return agent_executor


def test():
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    agent_executor = load_agent_executor(ci_params)
    test_input = "pythonで円周率を表示するプログラムを実行してください。"
    agent_executor_output = agent_executor.invoke({"input": test_input})
    print("agent_executor_output=", agent_executor_output)


if __name__ == "__main__":
    test()
