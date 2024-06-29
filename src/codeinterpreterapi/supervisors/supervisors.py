import getpass
import os
import platform
from typing import Union

from langchain.agents import AgentExecutor, AgentOutputParser, Tool
from langchain.schema import AgentAction, AgentFinish
from langchain_core.runnables import Runnable

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.prompts import create_supervisor_agent_prompt


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

        # prompt
        prompt = create_supervisor_agent_prompt(ci_params.is_ja)
        prompt = prompt.partial(options=str(options), members=", ".join(members))
        input_variables = prompt.input_variables
        print("choose_supervisor prompt.input_variables=", input_variables)

        # Using openai function calling can make output parsing easier for us
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }

        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                print("parse llm_output=", llm_output)
                if "Route" in llm_output:
                    next_action = llm_output.split("Route:")[-1].strip()
                    if next_action in options:
                        return AgentAction(tool="Route", tool_input=next_action, log=llm_output)
                    else:
                        return AgentFinish(
                            return_values={
                                "result": f"Invalid next action. Available options are: {', '.join(options)}"
                            },
                            log=llm_output,
                        )
                else:
                    return AgentFinish(
                        return_values={
                            "result": f"Agent did not select the Route tool. Available options are: {', '.join(options)}"
                        },
                        log=llm_output,
                    )

        output_parser = CustomOutputParser()

        # tool
        def route(next_action: str) -> str:
            if next_action not in options:
                return f"Invalid next action. Available options are: {', '.join(options)}"
            return f"The next action is: {next_action}"

        tool = Tool(
            name="Route",
            func=route,
            description="Select the next role from the available options.",
            schema=function_def["parameters"],
        )

        # agent
        supervisor_agent = prompt | ci_params.llm | output_parser
        ci_params.supervisor_agent = supervisor_agent

        # agent_executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=supervisor_agent, tools=[tool], verbose=ci_params.verbose
        )

        return agent_executor


def test():
    # sample = "ステップバイステップで2*5+2を計算して。"
    sample = "pythonで円周率を表示するプログラムを実行してください。"
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    supervisor = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)
    result = supervisor.invoke({"input": sample, "messages": [sample]})
    print("result=", result)


if __name__ == "__main__":
    test()
