from typing import Dict, List

from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from gui_agent_loop_core.schema.message.schema import BaseMessageContent
from langchain_core.prompts import PromptTemplate

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.crew.custom_agent import (
    CustomAgent,  # You need to build and extend your own agent logic with the CrewAI BaseAgent class then import it here.
)
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.schema import CodeInterpreterIntermediateResult, CodeInterpreterPlan, CodeInterpreterPlanList
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


class CodeInterpreterCrew:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.name_agent_dict = self.create_agents()
        self.agents = list(self.name_agent_dict.values())

    def create_agents(self) -> List[Agent]:
        name_agent_dict = {}
        # tools = AgentWrapperTool.create_agent_wrapper_tools(self.ci_params)
        for agent_def in self.ci_params.agent_def_list:
            agent_executor = agent_def.agent_executor
            role = agent_def.agent_name
            goal = "clear information provide for user about " + agent_def.agent_name
            backstory = agent_def.agent_role
            is_use_custom = True
            if is_use_custom:
                agent = CustomAgent(
                    agent_executor=agent_executor, ci_params=self.ci_params, role=role, goal=goal, backstory=backstory
                )
            else:
                # TODO: fix it. it is not working now.
                agent = Agent(
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    # tools=[tools[0]],
                    llm=self.ci_params.llm,
                )

            name_agent_dict[agent_def.agent_name] = agent
        return name_agent_dict

    def create_tasks(self, final_goal: str, plan_list: CodeInterpreterPlanList) -> List[Task]:
        tasks = []
        for plan in plan_list.agent_task_list:
            task = self.create_task(final_goal, plan)
            if task is not None:
                tasks.append(task)
        return tasks

    def create_task(self, final_goal: str, plan: CodeInterpreterPlan) -> Task:
        # find
        for agent_def in self.ci_params.agent_def_list:
            if plan.agent_name == agent_def.agent_name:
                task_description = "タスク（最終的なゴール）： " + final_goal
                task_description += (
                    "\n\nサブタスク（これを達成したら完了として処理終了してください）： " + plan.task_description
                )
                task = Task(
                    expected_output=plan.expected_output,
                    description=task_description,
                    agent=self.name_agent_dict[plan.agent_name],
                )
                return task

        print("WARN: no task found plan.agent_name=", plan.agent_name)
        return None

    def run(self, inputs: BaseMessageContent, plan_list: CodeInterpreterPlanList) -> CodeInterpreterIntermediateResult:
        # update task description
        if plan_list is None:
            return {}

        if isinstance(inputs, list):
            last_input = inputs[-1]
        else:
            last_input = inputs
        if "input" in inputs:
            final_goal = last_input["input"]
        elif "content" in inputs:
            final_goal = last_input["content"]
        else:
            final_goal = "ユーザの指示に従って最終的な回答をしてください"

        tasks = self.create_tasks(final_goal=final_goal, plan_list=plan_list)
        my_crew = Crew(agents=self.agents, tasks=tasks)
        crew_output: CrewOutput = my_crew.kickoff(inputs=last_input)
        result = self.llm_convert_to_CodeInterpreterIntermediateResult(crew_output, last_input, final_goal)
        return result

    def llm_convert_to_CodeInterpreterIntermediateResult(
        self, crew_output: CrewOutput, last_input: Dict, final_goal: str
    ) -> CodeInterpreterIntermediateResult:
        prompt_template = """
        CrewOutputクラスの内容をCodeInterpreterIntermediateResultクラスに詰め替えてください。
        作業目的を考慮して重要な情報から可能な限り詰めてください。
        不要な情報や重複する情報はdropしてください。
        取得できなかった変数はデフォルト値のままにしてください。

        ## 作業目的
        {final_goal}

        ## crew_output
        {crew_output}
        """

        prompt = PromptTemplate(
            input_variables=["final_goal", "agent_scratchpad", "crew_output"], template=prompt_template
        )

        structured_llm = self.ci_params.llm.with_structured_output(
            schema=CodeInterpreterIntermediateResult, include_raw=False
        )
        runnable = prompt | structured_llm

        last_input["final_goal"] = final_goal
        last_input["agent_scratchpad"] = crew_output.raw
        last_input["crew_output"] = crew_output.tasks_output
        output = runnable.invoke(input=last_input)
        return output


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    inputs = {"input": TestPrompt.svg_input_str}
    plan = CodeInterpreterPlan(agent_name="code_write_agent", task_description="", expected_output="")
    plan_list = CodeInterpreterPlanList(reliability=80, agent_task_list=[plan, plan])
    result = CodeInterpreterCrew(ci_params).run(inputs, plan_list)
    print(result)


if __name__ == "__main__":
    test()
