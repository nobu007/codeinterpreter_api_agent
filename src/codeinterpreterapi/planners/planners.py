from typing import List

from langchain import hub
from langchain.agents import create_react_agent
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain_core.runnables import Runnable


class CodeInterpreterPlanner:
    @staticmethod
    def choose_planner(llm: BaseLanguageModel, tools: List[BaseTool], is_ja: bool) -> Runnable:
        """
        Load a chat planner.

        Args:
            llm: Language model.
            tools: List of tools this agent has access to.
            is_ja: System prompt.

        Returns:
            LLMPlanner
        """

        prompt_name = "nobu/simple_react"
        if is_ja:
            prompt_name += "_ja"
        prompt = hub.pull(prompt_name)
        planner_agent = create_react_agent(llm, tools, prompt)
        return planner_agent
