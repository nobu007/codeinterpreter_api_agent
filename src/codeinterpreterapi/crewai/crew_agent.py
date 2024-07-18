from typing import Dict

from crewai import Agent, Crew, Task

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.crewai.custom_agent import (
    CustomAgent,  # You need to build and extend your own agent logic with the CrewAI BaseAgent class then import it here.
)
from codeinterpreterapi.graphs.agent_wrapper_tool import AgentWrapperTool
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


def run(ci_params: CodeInterpreterParams, inputs: Dict):
    agents = []
    tasks = []
    tools = AgentWrapperTool.create_agent_wrapper_tools(ci_params)
    for agent_def in ci_params.agent_def_list:
        agent_executor = agent_def.agent_executor
        role = agent_def.agent_name
        goal = "clear information provide for user about " + agent_def.agent_name
        backstory = agent_def.agent_role
        agent_custom = CustomAgent(agent_executor=agent_executor, role=role, goal=goal, backstory=backstory)
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[tools[0]],
            llm=ci_params.llm_fast,
        )
        agents.append(agent_custom)

        task = Task(
            expected_output=agent_def.agent_role + " of {input}",
            description=agent_def.agent_role,
            agent=agent,
        )
        tasks.append(task)

    my_crew = Crew(agents=agents, tasks=tasks)
    result = my_crew.kickoff(inputs=inputs)
    return result


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)
    inputs = {"input": TestPrompt.svg_input_str}
    result = run(ci_params, inputs)
    print(result)


if __name__ == "__main__":
    test()
