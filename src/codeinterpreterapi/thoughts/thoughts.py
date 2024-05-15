from gui_agent_loop_core.thoughts.prompts import get_propose_prompt, get_propose_prompt_ja
from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.pydantic_v1 import Field

# from langchain_experimental.tot.prompts import get_cot_prompt, get_propose_prompt
from langchain_experimental.tot.thought_generation import BaseThoughtGenerationStrategy, ProposePromptStrategy


class MyProposePromptStrategy(ProposePromptStrategy):
    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt)


class MyProposePromptStrategyJa(ProposePromptStrategy):
    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt_ja)
