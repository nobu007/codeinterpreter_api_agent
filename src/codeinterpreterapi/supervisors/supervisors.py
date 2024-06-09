import getpass
import os
import platform

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import Runnable

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
        if ci_params.is_ja:
            prompt_name += "_ja"
        # plan_agent = RunnableAgent(runnable=planner)
        # prompt = hub.pull("nobu/code_writer:0c56967d")
        input_variables = exec_prompt.input_variables
        print("choose_supervisor prompt.input_variables=", input_variables)

        # exec_agent
        # exec_prompt = hub.pull("nobu/code_writer:0c56967d")
        exec_prompt = hub.pull("hwchase17/openai-tools-agent")
        # exec_prompt = create_complement_input(exec_prompt) | exec_prompt
        # exec_runnable = exec_prompt | ci_params.llm_fast | CustomOutputParser()
        # remapped_inputs = create_complement_input(exec_prompt).invoke({})
        # exec_agent = RunnableAgent(runnable=exec_runnable, input_keys=list(remapped_inputs.keys()))
        exec_agent = create_tool_calling_agent(ci_params.llm_switcher, ci_params.tools, exec_prompt)

        # plan_chain
        agent_executor = AgentExecutor(agent=exec_agent, tools=ci_params.tools, verbose=ci_params.verbose)

        return agent_executor


def test():
    sample = "ステップバイステップで2*5+2を計算して。"
    llm = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    supervisor = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)
    result = supervisor.invoke({"input": sample, "agent_scratchpad": ""})
    print("result=", result)


if __name__ == "__main__":
    test()
