import getpass
import os
import platform
from typing import Any, Dict

from langchain.agents import AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.crew.crew_agent import CodeInterpreterCrew
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.schema import CodeInterpreterPlanList
from codeinterpreterapi.supervisors.prompts import create_supervisor_agent_prompt
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


class CodeInterpreterSupervisor:
    def __init__(self, planner: Runnable, ci_params: CodeInterpreterParams):
        self.planner = ci_params.planner_agent
        self.ci_params = ci_params
        self.supervisor_chain = None
        self.initialize()

    def initialize(self) -> None:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)

        # options
        members = []
        for agent_def in self.ci_params.agent_def_list:
            members.append(agent_def.agent_name)

        options = ["FINISH"] + members
        print("options=", options)

        # prompt(for executor)
        prompt = create_supervisor_agent_prompt(self.ci_params.is_ja)
        prompt = prompt.partial(options=str(options), members=", ".join(members))
        input_variables = prompt.input_variables
        print("choose_supervisor prompt.input_variables=", input_variables)

        class RouteSchema(BaseModel):
            next: str = Field(..., description=f"The next route item. This is one of: {options}")
            # question: str = Field(..., description="The original question from user.")

        # agent
        # TODO: use RouteSchema to determine use crew or agent or agent executor
        # llm_with_structured_output = self.ci_params.llm.with_structured_output(RouteSchema)
        supervisor_agent = prompt | self.ci_params.llm
        # supervisor_agent_for_executor = prompt | ci_params.llm
        # self.supervisor_chain = self.planner | prompt | llm_with_structured_output
        self.supervisor_chain = self.planner
        self.ci_params.supervisor_agent = supervisor_agent

        # config
        # if self.ci_params.runnable_config:
        #     self.supervisor_chain = self.supervisor_chain.with_config(self.ci_params.runnable_config)

    def get_executor(self) -> AgentExecutor:
        # TODO: use own executor(not crewai)
        # agent_executor
        # agent_executor = AgentExecutor.from_agent_and_tools(
        #     agent=supervisor_agent_for_executor,
        #     tools=[tool],
        #     verbose=ci_params.verbose,
        #     callbacks=[],
        # )
        # TODO: impl
        return self.supervisor_chain

    def invoke(self, input: Input) -> Output:
        result = self.planner.invoke(input)
        print("type result=", type(result))
        result = self.ci_params.crew_agent.run(input, result)
        return result

    def execute_plan(self, plan_list: CodeInterpreterPlanList) -> Dict[str, Any]:
        # AgentExecutorの初期化
        agent = self.get_executor()

        results = []

        for plan in plan_list.agent_task_list:
            print(f"\nExecuting task: {plan.task_description}")
            print(f"Agent: {plan.agent_name}")
            print(f"Expected output: {plan.expected_output}")

            if plan.agent_name == "<END_OF_PLAN>":
                print("Reached end of plan. Execution complete.")
                break

            try:
                # タスクを実行
                result = agent.run(
                    f"Task: {plan.task_description}\n"
                    f"Expected output: {plan.expected_output}\n"
                    "Please complete this task."
                )

                results.append({"task": plan.task_description, "agent": plan.agent_name, "result": result})

                print(f"Task result: {result}")

            except Exception as e:
                print(f"Error executing task: {e}")
                results.append({"task": plan.task_description, "agent": plan.agent_name, "error": str(e)})

        return {
            "plan_execution_results": results,
        }


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    crew_agent = CodeInterpreterCrew(ci_params=ci_params)
    ci_params.crew_agent = crew_agent
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    supervisor = CodeInterpreterSupervisor(planner=planner, ci_params=ci_params)
    result = supervisor.invoke(
        {"input": TestPrompt.svg_input_str, "agent_scratchpad": "", "messages": [TestPrompt.svg_input_str]}
    )
    print("result=", result)


if __name__ == "__main__":
    test()
