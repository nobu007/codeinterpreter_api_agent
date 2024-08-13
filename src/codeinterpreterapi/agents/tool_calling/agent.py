from typing import Sequence

from langchain.agents.agent import AgentOutputParser
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from langchain_core.tools import BaseTool

from codeinterpreterapi.agents.tool_calling.prompts import create_tool_calling_agent_prompt
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.utils.runnable import create_complement_input
from codeinterpreterapi.utils.runnable_history import assign_runnable_history


def create_tool_calling_agent_wrapper(
    ci_params: CodeInterpreterParams,
    prompt: ChatPromptTemplate,
) -> Runnable:
    return create_tool_calling_agent(llm=ci_params.llm_tools, tools=ci_params.tools, prompt=prompt)


def create_tool_calling_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    output_parser: AgentOutputParser = ToolsAgentOutputParser(),
    runnable_config: RunnableConfig = RunnableConfig(callbacks=None, tags=[], metadata={}),
) -> Runnable:
    """Create an agent that uses tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            model = ChatAnthropic(model="claude-3-opus-20240229")

            @tool
            def magic_function(input: int) -> int:
                \"\"\"Applies a magic function to an input.\"\"\"
                return input + 2

            tools = [magic_function]

            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.
    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables + list(prompt.partial_variables))
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError(
            "This function requires a .bind_tools method be implemented on the LLM.",
        )
    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]))
        | create_complement_input(prompt)
        | prompt
        | llm_with_tools
    )
    agent = assign_runnable_history(agent, runnable_config)
    agent = agent | output_parser
    return agent


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    prompt = create_tool_calling_agent_prompt(ci_params.is_ja)
    agent = create_tool_calling_agent_wrapper(ci_params=ci_params, prompt=prompt)
    test_input = "pythonで円周率を表示するプログラムを実行してください。"
    agent_output = agent.invoke({"input": test_input, "intermediate_steps": []})
    print("agent_output=", agent_output)


if __name__ == "__main__":
    test()
