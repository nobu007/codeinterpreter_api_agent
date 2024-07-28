from typing import List

from langchain import PromptTemplate
from pydantic import BaseModel, Field

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt
from codeinterpreterapi.tools.tools import CodeInterpreterTools
from codeinterpreterapi.utils.runnable import create_complement_input


class TestAgentPlan(BaseModel):
    '''Plan for the task.'''

    agent_name: str = Field(description="The agent name for task")
    agent_role: str = Field(description="What should do the agent")


class TestAgentPlanList(BaseModel):
    '''Plan for the task.'''

    agent_plan_list: List[TestAgentPlan] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


def tool_run(ci_params: CodeInterpreterParams, inputs: dict):
    prompt = get_prompt(ci_params)
    structured_llm = ci_params.llm_smart.bind_tools(tools=[TestAgentPlanList])
    runnable = (
        create_complement_input(prompt) | prompt | structured_llm
        # | StrOutputParser()
        # | PlannerSingleOutputParser()
    )
    return runnable.invoke(inputs)


def structured_run(ci_params: CodeInterpreterParams, inputs: dict):
    prompt = get_prompt(ci_params)
    structured_llm = ci_params.llm_smart.with_structured_output(TestAgentPlanList)
    runnable = (
        create_complement_input(prompt) | prompt | structured_llm
        # | StrOutputParser()
        # | PlannerSingleOutputParser()
    )
    return runnable.invoke(inputs)


def get_prompt(ci_params: CodeInterpreterParams):
    prompt_template = """
    あなたは経験豊富なプロジェクトマネージャーです。
    新しいAI関連のプロジェクトで以下のAIを利用したシーケンシャルな作業計画を作成してください。

    制約条件:
    - 段階的に考え、各ステップで取るアクションを明確にすること。
    - 必要な情報が不足している場合は、agent_name=Noneで回答し、agent_roleでユーザーに質問すること。
    - 実行に必要なagentが存在しない場合は、agent_name=Noneで回答し、agent_roleで必要なagentについて回答すること。
    - タスクを完了するために十分な情報が得られたら、最終的な作業計画を出力すること。
    - 最後のステップはagent_name=<END_OF_PLAN>とすること。

    利用可能なAI agentのリスト:
    {agent_info}

    回答内容:
    - 作業の順番を示したCodeInterpreterPlanListを回答してください。

    タスク: {input}

    agent_scratchpad: {agent_scratchpad}
    """
    prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=prompt_template)

    # 利用可能なエージェントの情報を文字列として作成
    agent_info_str = "\n".join([agent_def.get_agent_info() for agent_def in ci_params.agent_def_list])
    print("agent_info_str=", agent_info_str)

    # agent_infoを部分的に適用
    prompt = prompt.partial(agent_info=agent_info_str)

    print("prompt=", prompt.template)
    return prompt


def test():
    llm, llm_tools, runnable_config = prepare_test_llm(is_smart=False)
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    ci_params.tools = CodeInterpreterTools(ci_params).get_all_tools()
    result = structured_run(ci_params=ci_params, inputs={"input": TestPrompt.svg_input_str, "agent_scratchpad": ""})
    print("result=", result)


if __name__ == "__main__":
    test()
