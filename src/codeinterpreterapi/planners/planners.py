from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.runnables import Runnable

from codeinterpreterapi.brain.params import CodeInterpreterParams


class CodeInterpreterPlanner:
    @staticmethod
    def choose_planner(ci_params: CodeInterpreterParams) -> Runnable:
        """
        Load a chat planner.

        Args:
            llm: Language model.
            tools: List of tools this agent has access to.
            is_ja: System prompt.

        Returns:
            LLMPlanner

        <prompt: simple_react>
        Input
          tools:
          tool_names:
          input:
          agent_scratchpad:
        Output
          content: Free text in str.
        """
        prompt_name = "nobu/simple_react"
        if ci_params.is_ja:
            prompt_name += "_ja"
        prompt = hub.pull(prompt_name)
        planner_agent = create_react_agent(ci_params.llm_fast, ci_params.tools, prompt)
        return planner_agent
