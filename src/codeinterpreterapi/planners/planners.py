from typing import List, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.prompts import create_planner_agent_chat_prompt, create_planner_agent_prompt
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt
from codeinterpreterapi.utils.prompt import PromptUpdater
from codeinterpreterapi.utils.runnable import create_complement_input


class CodeInterpreterPlan(BaseModel):
    '''Agent and Task definition. Plan and task is 1:1.'''

    agent_name: str = Field(
        description="The agent name for task. This is primary key. Agent responsible for task execution. Represents entity performing task."
    )
    task_description: str = Field(description="Descriptive text detailing task's purpose and execution.")
    expected_output: str = Field(description="Clear definition of expected task outcome.")


class CodeInterpreterPlanList(BaseModel):
    '''Sequential plans for the task.'''

    agent_task_list: List[CodeInterpreterPlan] = Field(
        description="The list of CodeInterpreterPlan. It means agent name and so on."
    )


class CodeInterpreterPlanner:
    @staticmethod
    def choose_planner(ci_params: CodeInterpreterParams) -> Union[Runnable, AgentExecutor]:
        """
        Load a chat planner.

        """

        is_chat_prompt = True
        if is_chat_prompt:
            prompt = create_planner_agent_chat_prompt()
            prompt = PromptUpdater.update_and_show_chat_prompt(prompt, ci_params)
        else:
            prompt = create_planner_agent_prompt()
            prompt = PromptUpdater.update_prompt(prompt, ci_params)
            PromptUpdater.show_prompt(prompt)

        # structured_llm
        structured_llm = ci_params.llm.bind_tools(tools=[CodeInterpreterPlanList])

        # runnable
        runnable = (
            create_complement_input(prompt) | prompt | structured_llm
            # | StrOutputParser()
            # | PlannerSingleOutputParser()
        )
        # runnable = assign_runnable_history(runnable, ci_params.runnable_config)

        # agent
        # planner_agent = create_react_agent(ci_params.llm_fast, ci_params.tools, prompt)
        print("choose_planner prompt.input_variables=", prompt.input_variables)
        remapped_inputs = create_complement_input(prompt).invoke({})
        agent = RunnableAgent(runnable=runnable, input_keys=list(remapped_inputs.keys()))

        # return executor or runnable
        return_as_executor = False
        if return_as_executor:
            # TODO: handle step by step by original OutputParser
            agent_executor = AgentExecutor(agent=agent, tools=ci_params.tools, verbose=ci_params.verbose)
            return agent_executor

        else:
            return runnable


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    result = planner.invoke({"input": "", "agent_scratchpad": "", "messages": [TestPrompt.svg_input_str]})
    print("result=", result)


if __name__ == "__main__":
    test()
