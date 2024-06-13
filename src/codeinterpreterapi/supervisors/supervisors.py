import getpass
import os
import platform

from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable

from codeinterpreterapi.agents.tool_calling.create_tool_calling_agent import create_tool_calling_agent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(planner: Runnable, ci_params: CodeInterpreterParams) -> AgentExecutor:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)

        # plan
        prompt_name = "nobu/code_writer"
        if ci_params.is_ja:
            prompt_name += "_ja"
        exec_prompt = hub.pull(prompt_name)

        # runnable_config
        runnable_config = ci_params.runnable_config

        # TODO: use plan agent
        # plan_agent = RunnableAgent(runnable=planner)
        input_variables = exec_prompt.input_variables
        print("choose_supervisor prompt.input_variables=", input_variables)

        # exec_agent
        # exec_prompt = hub.pull("hwchase17/openai-tools-agent")
        exec_agent = create_tool_calling_agent(ci_params.llm_tools, ci_params.tools, exec_prompt, runnable_config)

        # agent_executor
        agent_executor = AgentExecutor(agent=exec_agent, tools=ci_params.tools, verbose=ci_params.verbose)

        return agent_executor


def test():
    # sample = "ステップバイステップで2*5+2を計算して。"
    sample = "pythonで円周率を表示するプログラムを実行してください。"
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    supervisor = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)
    result = supervisor.invoke({"input": sample})
    print("result=", result)


if __name__ == "__main__":
    test()
