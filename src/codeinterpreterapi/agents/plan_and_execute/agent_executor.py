from typing import List, Optional

from langchain.agents.agent import AgentExecutor, AgentOutputParser
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.plan_and_execute.executors.base import ChainExecutor

from codeinterpreterapi.agents.plan_and_execute.prompts import (
    FORMAT_INSTRUCTIONS,
    FORMAT_INSTRUCTIONS_JA,
    HUMAN_MESSAGE_TEMPLATE,
    SUFFIX,
    SUFFIX_JA,
    TASK_PREFIX,
    TOOLS_PREFIX,
    TOOLS_PREFIX_JA,
)


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
    input_variables = ["previous_steps", "current_step", "agent_scratchpad"]

    # message_template
    message_template = ""
    if include_task_in_prompt:
        input_variables.append("objective")
        message_template += TASK_PREFIX
    message_template += HUMAN_MESSAGE_TEMPLATE

    # format_instructions, tools_prefix, suffix
    if is_ja:
        format_instructions = FORMAT_INSTRUCTIONS_JA
        tools_prefix = TOOLS_PREFIX_JA
        suffix = SUFFIX_JA
    else:
        format_instructions = FORMAT_INSTRUCTIONS
        tools_prefix = TOOLS_PREFIX
        suffix = SUFFIX
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        callback_manager=callback_manager,
        output_parser=output_parser,
        prefix=tools_prefix,
        suffix=suffix,
        human_message_template=message_template,
        format_instructions=format_instructions,
        input_variables=input_variables,
        # memory_prompts = memory_prompts,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
    return ChainExecutor(chain=agent_executor)
