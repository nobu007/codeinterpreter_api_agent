from langchain import PromptTemplate, hub
from langchain.agents import AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain.tools.render import render_text_description
from langchain_core.runnables import Runnable

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.utils.output_parser import PlannerSingleOutputParser
from codeinterpreterapi.utils.runnable import create_complement_input


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
        # prompt_name_react = "nobu/simple_react"
        prompt_name = "nobu/chat_planner"
        if ci_params.is_ja:
            prompt_name += "_ja"
        prompt = hub.pull(prompt_name)
        # prompt = CodeInterpreterPlanner.get_prompt()

        # runnable
        runnable = (
            create_complement_input(prompt)
            | prompt
            | ci_params.llm
            # | StrOutputParser()
            | PlannerSingleOutputParser()
        )
        # runnable = assign_runnable_history(runnable, ci_params.runnable_config)

        # agent
        # planner_agent = create_react_agent(ci_params.llm_fast, ci_params.tools, prompt)
        print("choose_planner prompt.input_variables=", prompt.input_variables)
        remapped_inputs = create_complement_input(prompt).invoke({})
        agent = RunnableAgent(runnable=runnable, input_keys=list(remapped_inputs.keys()))

        # agent_executor
        agent_executor = AgentExecutor(agent=agent, tools=[], verbose=ci_params.verbose)

        return agent_executor

    @staticmethod
    def update_prompt(prompt: PromptTemplate, ci_params: CodeInterpreterParams) -> PromptTemplate:
        # Check if the prompt has the required input variables
        missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
            prompt.input_variables + list(prompt.partial_variables)
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        # Partial the prompt with tools and tool_names
        prompt = prompt.partial(
            tools=render_text_description(list(ci_params.tools)),
            tool_names=", ".join([t.name for t in ci_params.tools]),
        )
        return prompt

    @staticmethod
    def get_prompt():
        prompt_template = """
    あなたは親切なアシスタントです。以下の制約条件を守ってタスクを完了させてください。

    制約条件:
    - 段階的に考え、各ステップで取るアクションを明確にすること。
    - 必要な情報が不足している場合は、ユーザーに質問すること。
    - タスクを完了するために十分な情報が得られたら、最終的な回答を出力すること。

    タスク: {input}
    agent_scratchpad: {agent_scratchpad}
    """
        prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=prompt_template)
        return prompt


def test():
    sample = "ステップバイステップで2*5+2を計算して。"
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    result = planner.invoke({"input": sample, "agent_scratchpad": ""})
    print("result=", result)


if __name__ == "__main__":
    test()
