import getpass
import os
import platform
from typing import Any, List, Union

from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.prompts import create_supervisor_agent_prompt
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(planner: Runnable, ci_params: CodeInterpreterParams) -> AgentExecutor:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)

        # options
        members = []
        for agent_def in ci_params.agent_def_list:
            members.append(agent_def.agent_name)

        options = ["FINISH"] + members
        print("options=", options)

        # prompt
        prompt = create_supervisor_agent_prompt(ci_params.is_ja)
        prompt = prompt.partial(options=str(options), members=", ".join(members))
        input_variables = prompt.input_variables
        print("choose_supervisor prompt.input_variables=", input_variables)

        # Using openai function calling can make output parsing easier for us
        # function_def = {
        #     "name": "route",
        #     "description": "Select the next role.",
        #     "parameters": {
        #         "title": "routeSchema",
        #         "type": "object",
        #         "properties": {
        #             "next": {
        #                 "title": "Next",
        #                 "anyOf": [
        #                     {"enum": options},
        #                 ],
        #             },
        #             "question": {
        #                 "title": "Question",
        #                 "type": "string",
        #                 "description": "もともとの質問内容",
        #             },
        #         },
        #         "required": ["next", "question"],
        #     },
        # }

        class RouteSchema(BaseModel):
            next: str = Field(..., description=f"The next route item. This is one of: {options}")
            question: str = Field(..., description="The original question from user.")

        class CustomOutputParserForGraph(AgentOutputParser):
            def parse(self, text: str) -> dict:
                print("CustomOutputParserForGraph parse text=", text)
                if isinstance(text, str):
                    next_route = text
                    next_route = next_route.strip()
                    next_route = next_route.replace("'", "")
                    next_route = next_route.replace('"', "")
                else:
                    next_route = "FINISH"
                return_values = {
                    "next": next_route,
                    "question": "",
                    "messages": [],
                    "intermediate_steps": [],
                }
                return return_values

        class CustomOutputParserForExecutor(CustomOutputParserForGraph):
            def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
                print("CustomOutputParserForExecutor parse text=", text)
                return_values = super().parse(text)
                return AgentFinish(
                    return_values=return_values,
                    log=text,
                )

        class CustomOutputParserForGraphPydantic(PydanticOutputFunctionsParser):
            def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
                print("CustomOutputParserForGraphPydantic parse_result result=", result)
                result = super().parse_result(result, partial)
                print("CustomOutputParserForGraphPydantic parse_result result after=", result)
                if isinstance(result, str):
                    next_route = result
                    next_route = next_route.strip()
                    next_route = next_route.replace("'", "")
                    next_route = next_route.replace('"', "")
                else:
                    next_route = "FINISH"
                return_values = {
                    "next": next_route,
                    "question": "",
                    "messages": [],
                    "intermediate_steps": [],
                }
                return return_values

        # output_parser = CustomOutputParserForGraph()
        # output_parser_for_executor = CustomOutputParserForExecutor()
        # output_parser_pydantic = CustomOutputParserForGraphPydantic(pydantic_schema=RouteSchema)

        # tool
        def route(next_action: str) -> str:
            if next_action not in options:
                return f"Invalid next action. Available options are: {', '.join(options)}"
            return f"The next action is: {next_action}"

        # tool = Tool(
        #     name="Route",
        #     func=route,
        #     description="Select the next role from the available options.",
        #     # schema=function_def["parameters"],
        # )

        # config = RunnableConfig({'callbacks': [StdOutCallbackHandler()]})

        # agent
        llm_with_structured_output = ci_params.llm.with_structured_output(RouteSchema)
        # supervisor_agent = prompt | ci_params.llm | output_parser
        # supervisor_agent_for_executor = prompt | ci_params.llm
        supervisor_agent_structured_output = prompt | llm_with_structured_output

        ci_params.supervisor_agent = supervisor_agent_structured_output

        # agent_executor
        # agent_executor = AgentExecutor.from_agent_and_tools(
        #     agent=supervisor_agent_for_executor,
        #     tools=[tool],
        #     verbose=ci_params.verbose,
        #     callbacks=[],
        # )

        return supervisor_agent_structured_output


def test():
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    supervisor = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)
    result = supervisor.invoke({"input": TestPrompt.svg_input_str, "messages": [TestPrompt.svg_input_str]})
    print("result=", result)


if __name__ == "__main__":
    test()
