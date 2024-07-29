from typing import List, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain.output_parsers import OutputFixingParser
from langchain.schema import AIMessage, Generation
from langchain_core.output_parsers import PydanticOutputParser
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


class Metadata(BaseModel):
    id: str = Field(description="The ID of the content")
    name: str = Field(description="The name of the content")
    type: str = Field(description="The type of the content")


class ResponseMetadata(BaseModel):
    id: str = Field(description="The ID of the response")
    model: str = Field(description="The model used for the response")
    stop_reason: str = Field(description="The reason for stopping")
    stop_sequence: str | None = Field(description="The stop sequence, if any")
    usage: dict = Field(description="The token usage information")


class CodeInterpreterPlanOutput(BaseModel):
    agent_task_list: List[CodeInterpreterPlan] = Field(description="List of agent tasks")
    metadata: Metadata = Field(description="Metadata of the content")
    response_metadata: ResponseMetadata = Field(description="Metadata of the response")


class CustomPydanticOutputParser(PydanticOutputParser):
    def parse(self, text) -> CodeInterpreterPlanList:
        print("parse text=", text)
        input_data = self.preprocess_input(text)
        return super().parse(input_data)

    def preprocess_input(self, input_data) -> str:
        if isinstance(input_data, AIMessage):
            return input_data.content
        elif isinstance(input_data, Generation):
            return input_data.text
        elif isinstance(input_data, str):
            return input_data
        else:
            raise ValueError(f"Unexpected input type: {type(input_data)}")


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

        # parser
        parser = CustomPydanticOutputParser(pydantic_object=CodeInterpreterPlanList)
        new_parser = OutputFixingParser.from_llm(parser=parser, llm=ci_params.llm)
        # runnable
        runnable = create_complement_input(prompt) | prompt | structured_llm | new_parser

        # agent
        agent = RunnableAgent(runnable=runnable, input_keys=list(prompt.input_variables))

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
