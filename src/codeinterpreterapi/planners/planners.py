from langchain.base_language import BaseLanguageModel
from langchain_experimental.plan_and_execute import load_chat_planner
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner


class CodeInterpreterPlanner:
    @staticmethod
    def choose_planner(
        llm: BaseLanguageModel,
    ) -> LLMPlanner:
        planner = load_chat_planner(llm)
        return planner
