from typing import Dict, List

from crewai import Agent, Crew, Task

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.crew.custom_agent import (
    CustomAgent,  # You need to build and extend your own agent logic with the CrewAI BaseAgent class then import it here.
)
from codeinterpreterapi.graphs.agent_wrapper_tool import AgentWrapperTool
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlan, CodeInterpreterPlanList, CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


class CodeInterpreterCrew:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.name_agent_dict = self.create_agents(ci_params)
        self.ci_params = ci_params
        self.agents = list(self.name_agent_dict.values())

    def create_agents(self, ci_params: CodeInterpreterParams) -> List[Agent]:
        name_agent_dict = {}
        tools = AgentWrapperTool.create_agent_wrapper_tools(ci_params)
        for agent_def in ci_params.agent_def_list:
            agent_executor = agent_def.agent_executor
            role = agent_def.agent_name
            goal = "clear information provide for user about " + agent_def.agent_name
            backstory = agent_def.agent_role
            is_use_custom = True
            if is_use_custom:
                agent = CustomAgent(agent_executor=agent_executor, role=role, goal=goal, backstory=backstory)
            else:
                # TODO: fix it. it is not working now.
                agent = Agent(
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    tools=[tools[0]],
                    llm=ci_params.llm_fast,
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

    def run(self, inputs: Dict, plan_list: CodeInterpreterPlanList):
        # update task description
        tasks = self.create_tasks(final_goal=inputs["input"], plan_list=plan_list)
        my_crew = Crew(agents=self.agents, tasks=tasks)
        result = my_crew.kickoff(inputs=inputs)
        return result


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor(planner=planner, ci_params=ci_params)
    inputs = {"input": TestPrompt.svg_input_str}
    plan = CodeInterpreterPlan(agent_name="main_function_create_agent", task_description="", expected_output="")
    plan_list = CodeInterpreterPlanList(agent_task_list=[plan])
    result = CodeInterpreterCrew(ci_params).run(inputs, plan_list)
    print(result)


if __name__ == "__main__":
    test()
