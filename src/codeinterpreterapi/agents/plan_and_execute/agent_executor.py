from typing import List, Optional

from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.structured_chat.base import create_structured_chat_agent
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.plan_and_execute.executors.base import ChainExecutor

from codeinterpreterapi.agents.plan_and_execute.prompts import create_structured_chat_agent_prompt


def load_agent_executor(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[AgentOutputParser] = None,
    verbose: bool = False,
    include_task_in_prompt: bool = False,
    is_ja: str = True,
) -> ChainExecutor:
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
    input_variables = ["previous_steps", "current_step", "agent_scratchpad", "tools", "tool_names"]
    print("input_variables=", input_variables)
    prompt = create_structured_chat_agent_prompt(is_ja)
    print("prompt=", prompt.get_prompts())
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        # callback_manager=callback_manager,
        # output_parser=output_parser,
        # prefix=tools_prefix,
        # suffix=suffix,
        prompt=prompt,
        # format_instructions=format_instructions,
        # input_variables=input_variables,
        # memory_prompts = memory_prompts,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
    return agent_executor
    # return ChainExecutor(chain=agent_executor, verbose=verbose)
