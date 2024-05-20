import getpass
import os
import platform

from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableAssign, RunnablePassthrough
from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner


class MySupervisorChain(Chain):
    pass


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(planner: LLMPlanner, executor: AgentExecutor, verbose: bool = False) -> MySupervisorChain:
        # prompt
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        print("choose_supervisor info=", info)

        supervisor = PlanAndExecute(planner=planner, executor=executor, verbose=verbose)
        # prompt = hub.pull("nobu/chat_planner")
        # agent = create_react_agent(llm, [], prompt)
        # return agent
        # prompt = hub.pull("nobu/code_writer:0c56967d")

        supervisor_chain = RunnablePassthrough() | supervisor
        return supervisor_chain

        supervisor_chain = RunnableAssign() | supervisor
        return supervisor_chain
