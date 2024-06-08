import traceback
from typing import Any, List, Optional, Union

from gui_agent_loop_core.schema.schema import AgentName
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


class CodeInterpreterBrain(Runnable):
    AGENT_SCORE_MAX = 100
    AGENT_SCORE_MIN = -100

    def __init__(self, ci_params: CodeInterpreterParams = CodeInterpreterParams()) -> None:
        self.ci_params = ci_params
        # codebox = CodeBox(requirements=settings.CUSTOM_PACKAGES)
        # self.ci_params.codebox = codebox
        self.ci_params.tools = CodeInterpreterTools(self.ci_params).get_all_tools()
        self.verbose = self.ci_params.verbose

        # brain logic vals
        self.current_agent: AgentName = AgentName.AGENT_EXECUTOR
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
        self.agent_executor = CodeInterpreterAgent.choose_agent_executor(ci_params=self.ci_params)

    def initialize_llm_planner(self):
        self.llm_planner = CodeInterpreterPlanner.choose_planner(ci_params=self.ci_params)

    def initialize_supervisor(self):
        self.supervisor = CodeInterpreterSupervisor.choose_supervisor(
            planner=self.llm_planner, ci_params=self.ci_params
        )

    def initialize_thought(self):
        self.thought = CodeInterpreterToT.get_runnable_tot_chain(ci_params=self.ci_params)

    def prepare_input(self, input: Input):
        ca = self.current_agent
        if ca == AgentName.AGENT_EXECUTOR:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        elif ca == AgentName.LLM_PLANNER:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        elif ca == AgentName.SUPERVISOR:
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        else:
            # ca == AgentName.THOUGHT
            if "intermediate_steps" in input:
                del input['intermediate_steps']
        return input

    def run(self, input: Input) -> Output:
        self.update_next_agent()
        input = self.prepare_input(input)
        print("Brain run self.current_agent=", self.current_agent)
        try:
            ca = self.current_agent
            if ca == AgentName.AGENT_EXECUTOR:
                output = self.agent_executor.invoke(input)
            elif ca == AgentName.LLM_PLANNER:
                output = self.llm_planner.invoke(input)
            elif ca == AgentName.SUPERVISOR:
                output = self.supervisor.invoke(input)
            else:
                # ca == AgentName.THOUGHT
                output = self.thought.invoke(input)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            output = "Error in CodeInterpreterSession: " f"{e.__class__.__name__}  - {e}"

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
        print("CodeInterpreterBrain agent_score=", self.current_agent_score, f"({self.current_agent})")

    def update_next_agent(self):
        if self.current_agent_score > CodeInterpreterBrain.AGENT_SCORE_MAX:
            self.current_agent_score = CodeInterpreterBrain.AGENT_SCORE_MAX
        if self.current_agent_score < CodeInterpreterBrain.AGENT_SCORE_MIN:
            self.current_agent_score = 0
            self.current_agent = self.use_next_agent()
        print("CodeInterpreterBrain update_next_agent=", self.current_agent)

    def use_next_agent(self):
        ca = self.current_agent
        if ca == AgentName.AGENT_EXECUTOR:
            return AgentName.LLM_PLANNER
        elif ca == AgentName.LLM_PLANNER:
            return AgentName.SUPERVISOR
        elif ca == AgentName.SUPERVISOR:
            return AgentName.THOUGHT
        else:
            # thought -> agent_executor
            return AgentName.AGENT_EXECUTOR

    def use_agent(self, new_agent: AgentName):
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
    brain.use_agent(AgentName.AGENT_EXECUTOR)
    # input_dict = {"input": "please exec print('test output')"}
    # result = brain.invoke(input_dict)
    # print("result=", result)
    # assert "test output" in result["output"]

    # # try2: llm_planner
    brain.use_agent(AgentName.LLM_PLANNER)
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
    brain.use_agent(AgentName.SUPERVISOR)
    result = brain.invoke(input_dict)
    print("result=", result)
    # assert (
    #     "test output" in result["output"]
    # )  # TODO: refine logic. now returns "Pythonコードを実行して出力を確認する。"

    # try4: thought
    input_dict = {
        "input": "please exec print('test output')",
    }
    brain.use_agent(AgentName.THOUGHT)
    result = brain.invoke(input_dict)
    print("result=", result)
    assert "test output" in result["output"]


if __name__ == "__main__":
    test()
