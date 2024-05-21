import getpass
import os
import platform
from typing import List

from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain_core.runnables import Runnable


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(
        planner: Runnable, executor: Runnable, tools: List[BaseTool], verbose: bool = False
    ) -> AgentExecutor:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)
        agent_executor = AgentExecutor(agent=planner, tools=tools, verbose=verbose)
        # prompt = hub.pull("nobu/chat_planner")
        # agent = create_react_agent(llm, [], prompt)
        # return agent
        # prompt = hub.pull("nobu/code_writer:0c56967d")

        return agent_executor
