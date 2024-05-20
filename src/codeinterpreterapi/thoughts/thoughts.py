from gui_agent_loop_core.thoughts.prompts import get_propose_prompt, get_propose_prompt_ja
from langchain.prompts.base import BasePromptTemplate
from langchain_experimental.pydantic_v1 import Field

# from langchain_experimental.tot.prompts import get_cot_prompt, get_propose_prompt
from langchain_experimental.tot.thought_generation import ProposePromptStrategy


class MyProposePromptStrategy(ProposePromptStrategy):
    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt)


class MyProposePromptStrategyJa(ProposePromptStrategy):
    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt_ja)
