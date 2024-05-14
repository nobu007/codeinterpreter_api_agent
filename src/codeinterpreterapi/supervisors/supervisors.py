from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner


class MySupervisorChain(Chain):
    pass


class CodeInterpreterSupervisor:
    @staticmethod
    def choose_supervisor(planner: LLMPlanner, executor: AgentExecutor, verbose: bool = False) -> MySupervisorChain:
        supervisor = PlanAndExecute(planner=planner, executor=executor, verbose=verbose)
        return supervisor
