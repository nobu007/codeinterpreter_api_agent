from typing import Any, List, Optional, Union

from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.thoughts.thoughts import CodeInterpreterToT
from codeinterpreterapi.tools.tools import CodeInterpreterTools


class AgentEnum:
    agent_executor = "agent_executor"
    llm_planner = "llm_planner"
    supervisor = "supervisor"
    thought = "thought"


class CodeInterpreterBrain(Runnable):
    AGENT_SCORE_MAX = 100
    AGENT_SCORE_MIN = -100

    def __init__(self, ci_params: CodeInterpreterParams = CodeInterpreterParams()) -> None:
        self.ci_params = ci_params
        # codebox = CodeBox(requirements=settings.CUSTOM_PACKAGES)
        # self.ci_params.codebox = codebox
        self.ci_params.tools = CodeInterpreterTools(self.ci_params).get_all_tools()

        # brain logic vals
        self.current_agent: AgentEnum = AgentEnum.agent_executor
        self.current_agent_score: int = 0

        # agents
        self.agent_executor: Optional[Runnable] = None
        self.llm_planner: Optional[Runnable] = None
        self.supervisor: Optional[AgentExecutor] = None
        self.thought: Optional[Runnable] = None

        # initialize agents
        self.initialize()

    def initialize(self):
        self.initialize_agent_executor()
        self.initialize_llm_planner()
        self.initialize_supervisor()
        self.initialize_thought()

    def initialize_agent_executor(self):
        is_experimental = True
        if is_experimental:
            self.agent_executor = CodeInterpreterAgent.create_agent_and_executor_experimental(ci_params=self.ci_params)
        else:
            self.agent_executor = CodeInterpreterAgent.create_agent_executor_lcel(ci_params=self.ci_params)

    def initialize_llm_planner(self):
        self.llm_planner = CodeInterpreterPlanner.choose_planner(ci_params=self.ci_params)

    def initialize_supervisor(self):
        self.supervisor = CodeInterpreterSupervisor.choose_supervisor(
            planner=self.llm_planner, ci_params=self.ci_params
        )

    def initialize_thought(self):
        self.thought = CodeInterpreterToT.get_runnable_tot_chain(ci_params=self.ci_params)

    def prepare_input(self, input: Input):
        if self.current_agent == AgentEnum.agent_executor:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        elif self.current_agent == AgentEnum.llm_planner:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        elif self.current_agent == AgentEnum.supervisor:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        else:
            # thought
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        return input

    def run(self, input: Input) -> Output:
        input = self.prepare_input(input)
        if self.current_agent == AgentEnum.agent_executor:
            output = self.agent_executor.invoke(input)
        elif self.current_agent == AgentEnum.llm_planner:
            output = self.llm_planner.invoke(input)
        elif self.current_agent == AgentEnum.supervisor:
            output = self.supervisor.invoke(input)
        else:
            # thought
            output = self.thought.invoke(input)
        self.update_agent_score()
        if isinstance(output, str):
            output = {"output": output}
        return output

    def __call__(self, input: Input) -> Output:
        return self.run(input)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.run(input)

    def batch(self, inputs: List[Output]) -> List[Output]:
        return [self.run(input_item) for input_item in inputs]

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        raise NotImplementedError("Async not implemented yet")

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        raise NotImplementedError("Async not implemented yet")

    def update_agent_score(self):
        self.current_agent_score = CodeInterpreterBrain.AGENT_SCORE_MIN - 1  # temp: switch every time
        if self.current_agent_score > CodeInterpreterBrain.AGENT_SCORE_MAX:
            self.current_agent_score = CodeInterpreterBrain.AGENT_SCORE_MAX
        if self.current_agent_score < CodeInterpreterBrain.AGENT_SCORE_MIN:
            self.current_agent_score = 0
            self.current_agent = self.use_next_agent()
        print("CodeInterpreterBrain agent_score=", self.current_agent_score, f"({self.current_agent})")

    def use_next_agent(self):
        current_agent = self.current_agent
        if current_agent == AgentEnum.agent_executor:
            return AgentEnum.llm_planner
        elif current_agent == AgentEnum.llm_planner:
            return AgentEnum.supervisor
        elif current_agent == AgentEnum.supervisor:
            return AgentEnum.thought
        else:
            # thought -> agent_executor
            return AgentEnum.agent_executor

    def use_agent(self, new_agent: AgentEnum):
        print("CodeInterpreterBrain use_agent=", new_agent)
        self.current_agent = new_agent


def test():
    settings.WORK_DIR = "/tmp"
    llm = prepare_test_llm()
    ci_params = CodeInterpreterParams(
        llm=llm,
        llm_fast=llm,
        llm_smart=llm,
        llm_local=llm,
        verbose=True,
    )
    brain = CodeInterpreterBrain(ci_params)

    # # try1: agent_executor
    brain.use_agent(AgentEnum.agent_executor)
    # input_dict = {"input": "please exec print('test output')"}
    # result = brain.invoke(input_dict)
    # print("result=", result)
    # assert "test output" in result["output"]

    # # try2: llm_planner
    brain.use_agent(AgentEnum.llm_planner)
    # input_dict = {
    #     "input": "please exec print('test output')",
    #     "intermediate_steps": "",
    # }
    # result = brain.invoke(input_dict)
    # print("result=", result)
    # assert "python" == result.tool
    # assert "test output" in result.tool_input

    # try3: supervisor
    input_dict = {
        "input": "please exec print('test output')",
    }
    brain.use_agent(AgentEnum.supervisor)
    result = brain.invoke(input_dict)
    print("result=", result)
    # assert (
    #     "test output" in result["output"]
    # )  # TODO: refine logic. now returns "Pythonコードを実行して出力を確認する。"

    # try4: thought
    input_dict = {
        "input": "please exec print('test output')",
    }
    brain.use_agent(AgentEnum.thought)
    result = brain.invoke(input_dict)
    print("result=", result)
    assert "test output" in result["output"]


if __name__ == "__main__":
    test()
