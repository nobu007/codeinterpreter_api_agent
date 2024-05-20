from langchain import hub
from langchain.agents import create_react_agent
from langchain.base_language import BaseLanguageModel
from langchain_core.runnables import Runnable

SYSTEM_PROMPT_PLANNER = """
Let's first understand the problem and devise a plan to solve the problem.
Please output the plan starting with the header 'Plan:'
and then followed by a numbered list of steps.
Please make the plan the minimum number of steps required
to accurately complete the task. If the task is a question,
the final step should almost always be 'Given the above steps taken,
please respond to the users original question'.
At the end of your plan, say '<END_OF_PLAN>'
"""

SYSTEM_PROMPT_PLANNER_JA = """
　まず、問題を理解し、問題を解決するための計画を立てましょう。
計画は'Plan:'の見出しで始め、番号付きのステップリストで出力してください。
タスクを正確に完了するために必要な最小限のステップ数で計画を立ててください。
タスクが質問である場合、最後のステップは通常、「上記の手順を踏まえて、ユーザーの元の質問に回答してください」となります。
計画の最後に'<END_OF_PLAN>'と記載してください。
"""


class CodeInterpreterPlanner:
    @staticmethod
    def choose_planner(llm: BaseLanguageModel, is_ja: bool) -> Runnable:
        """
        Load a chat planner.

        Args:
            llm: Language model.
            system_prompt: System prompt.

        Returns:
            LLMPlanner
        """
        system_prompt = SYSTEM_PROMPT_PLANNER_JA if is_ja else SYSTEM_PROMPT_PLANNER
        print("system_prompt(planner)=", system_prompt)
        prompt = hub.pull("nobu/chat_planner")
        planner_agent = create_react_agent(llm, [], prompt)
        return planner_agent
