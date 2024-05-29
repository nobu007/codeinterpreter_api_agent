import getpass
import os
import platform

from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable

from codeinterpreterapi.brain.params import CodeInterpreterParams


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(planner: Runnable, ci_params: CodeInterpreterParams) -> AgentExecutor:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)
        agent_executor = AgentExecutor(agent=planner, tools=ci_params.tools, verbose=ci_params.verbose)
        # prompt = hub.pull("nobu/chat_planner")
        # agent = create_react_agent(llm, [], prompt)
        # return agent
        # prompt = hub.pull("nobu/code_writer:0c56967d")

        return agent_executor
