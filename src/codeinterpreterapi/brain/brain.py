import random
import traceback
from typing import Any, Dict, List, Optional, Union

from gui_agent_loop_core.schema.core.schema import AgentName
from gui_agent_loop_core.schema.message.schema import BaseMessageContent
from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.config import settings
from codeinterpreterapi.crew.crew_agent import CodeInterpreterCrew
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.schema import CodeInterpreterIntermediateResult, CodeInterpreterPlanList
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.thoughts.thoughts import CodeInterpreterToT
from codeinterpreterapi.tools.tools import CodeInterpreterTools
from codeinterpreterapi.utils.multi_converter import MultiConverter


class CodeInterpreterBrain(Runnable):
    AGENT_SCORE_MAX = 100
    AGENT_SCORE_MIN = -100

    def __init__(self, ci_params: CodeInterpreterParams) -> None:
        self.ci_params = ci_params
        # codebox = CodeBox(requirements=settings.CUSTOM_PACKAGES)
        # self.ci_params.codebox = codebox
        self.ci_params.tools = CodeInterpreterTools(self.ci_params).get_all_tools()
        self.verbose = self.ci_params.verbose

        # brain logic vals
        self.current_agent: AgentName = AgentName.SUPERVISOR
        self.current_agent_score: int = 0

        # agents
        self.agent_executors: Optional[List[Runnable]] = []
        self.agent_executor: Optional[AgentExecutor] = None
        self.llm_planner: Optional[Runnable] = None
        self.supervisor: Optional[CodeInterpreterSupervisor] = None
        self.thought: Optional[Runnable] = None
        self.crew_agent: Optional[CodeInterpreterCrew] = None

        # agent_results
        self.agent_executor_result: Optional[str] = ""
        self.plan_list: Optional[CodeInterpreterPlanList] = None
        self.supervisor_result: Optional[CodeInterpreterIntermediateResult] = ""
        self.thought_result: Optional[str] = ""
        self.crew_result: Optional[str] = ""

        # initialize agents
        self.initialize()

    def initialize(self):
        self.initialize_agent_executor()
        self.initialize_crew()
        self.initialize_llm_planner()
        self.initialize_supervisor()
        self.initialize_thought()

    def initialize_agent_executor(self):
        self.agent_executors = CodeInterpreterAgent.choose_agent_executors(ci_params=self.ci_params)
        self.agent_executor = self.agent_executors[0]

    def initialize_llm_planner(self):
        self.llm_planner = CodeInterpreterPlanner.choose_planner(ci_params=self.ci_params)

    def initialize_supervisor(self):
        self.supervisor = CodeInterpreterSupervisor(planner=self.llm_planner, ci_params=self.ci_params)

    def initialize_thought(self):
        self.thought = CodeInterpreterToT.get_runnable_tot_chain(ci_params=self.ci_params)

    def initialize_crew(self):
        self.crew_agent = CodeInterpreterCrew(ci_params=self.ci_params)
        self.ci_params.crew_agent = self.crew_agent

    def prepare_input(self, input_dict: Dict):
        ca = self.current_agent
        if ca == AgentName.AGENT_EXECUTOR:
            if "intermediate_steps" in input_dict:
                del input_dict['intermediate_steps']
        elif ca == AgentName.LLM_PLANNER:
            if "intermediate_steps" in input_dict:
                del input_dict['intermediate_steps']
        elif ca == AgentName.SUPERVISOR:
            if "intermediate_steps" in input_dict:
                del input_dict['intermediate_steps']
            if "agent_scratchpad" not in input_dict:
                input_dict['agent_scratchpad'] = ""
            if "messages" not in input_dict:
                input_dict['messages'] = []
        elif ca == AgentName.THOUGHT:
            if "intermediate_steps" in input_dict:
                del input_dict['intermediate_steps']
        else:
            # ca == AgentName.CREW
            if "intermediate_steps" in input_dict:
                del input_dict['intermediate_steps']

        return input_dict

    def run(
        self, input: BaseMessageContent, runnable_config: Optional[RunnableConfig] = None
    ) -> CodeInterpreterIntermediateResult:
        self.update_next_agent()
        last_input = input
        if isinstance(input, list):
            last_input = input[-1]
            if isinstance(last_input, Dict):
                input[-1] = self.prepare_input(last_input)
        else:
            if isinstance(last_input, Dict):
                input = self.prepare_input(last_input)
        print("Brain run self.current_agent=", self.current_agent)
        print("Brain run input=", input)
        try:
            ca = self.current_agent
            if ca == AgentName.AGENT_EXECUTOR:
                # TODO: set output
                self.agent_executor_result = self.agent_executor.invoke(input, config=runnable_config)
                result = self.agent_executor_result

            elif ca == AgentName.LLM_PLANNER:
                # TODO: set output
                self.plan_list = self.llm_planner.invoke(input)
                result = self.plan_list
            elif ca == AgentName.SUPERVISOR:
                self.supervisor_result = self.supervisor.invoke(input)
                result = self.supervisor_result
            elif ca == AgentName.THOUGHT:
                # TODO: fix it and set output
                self.thought_result = self.thought.invoke(input, config=runnable_config)
                result = self.thought_result
            else:
                # ca == AgentName.CREW
                self.crew_result = self.crew_agent.run(input, self.plan_list)
                result = self.crew_result
            output = self._set_output_llm_result(result)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            context = "Error in CodeInterpreterSession: " f"{e.__class__.__name__}  - {e}"
            output = CodeInterpreterIntermediateResult(context=context)

        self.update_agent_score()
        if isinstance(output, str):
            output = CodeInterpreterIntermediateResult(context=output)
        return output

    def _set_output_llm_result(
        self, result: Union[Dict[str, Any], CodeInterpreterIntermediateResult, CodeInterpreterPlanList]
    ) -> CodeInterpreterIntermediateResult:
        output = CodeInterpreterIntermediateResult(context="")
        if isinstance(result, Dict):
            output = MultiConverter.to_CodeInterpreterIntermediateResult(result)
        elif isinstance(result, CodeInterpreterIntermediateResult):
            return result
        elif isinstance(result, CodeInterpreterPlanList):
            plan_list: CodeInterpreterPlanList = result
            output.context = str(plan_list)
        else:
            output.context = str(result)
            output.log = f"type={type(result)}"

        return output

    def __call__(self, input: Input) -> Output:
        return self.run(input)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.run(input, config)

    def batch(self, inputs: List[Output]) -> List[Output]:
        return [self.run(input_item) for input_item in inputs]

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        return self.run(input, config)

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        return [self.run(input_item) for input_item in inputs]

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
            # always use SUPERVISOR
            return AgentName.SUPERVISOR
        elif ca == AgentName.THOUGHT:
            self.agent_executor = random.choice(self.agent_executors)
            return AgentName.AGENT_EXECUTOR
        else:
            # CREW -> CREW
            return AgentName.CREW

    def use_agent(self, new_agent: AgentName):
        print("CodeInterpreterBrain use_agent=", new_agent)
        self.current_agent = new_agent


def test():
    settings.WORK_DIR = "/tmp"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor(planner=planner, ci_params=ci_params)

    ci_params.tools = []
    ci_params.tools = CodeInterpreterTools(ci_params).get_all_tools()
    brain = CodeInterpreterBrain(ci_params)

    if True:
        # try1: agent_executor
        print("try1: agent_executor")
        brain.use_agent(AgentName.AGENT_EXECUTOR)
        # sample = "ツールのpythonで円周率を表示するプログラムを実行してください。"
        sample = "please exec print('test output')"
        input_dict = {"input": sample}
        result = brain.invoke(input=input_dict)
        print("result=", result)
        assert "test output" in result.context

    if False:
        # try2: llm_planner
        print("try2: llm_planner")
        brain.use_agent(AgentName.LLM_PLANNER)
        input_dict = {
            "input": "please exec print('test output')",
            "intermediate_steps": "",
        }
        result = brain.invoke(input_dict)
        print("result=", result)
        assert "python" == result.tool
        assert "test output" in result.tool_input

    if False:
        # try3: supervisor
        print("try3: supervisor")
        sample = "ステップバイステップで2*5+2を計算して。"
        input_dict = {
            "input": sample,
        }
        brain.use_agent(AgentName.SUPERVISOR)
        result = brain.invoke(input=input_dict, runnable_config=runnable_config)
        print("result=", result)
        # assert (
        #     "test output" in result["output"]
        # )  # TODO: refine logic. now returns "Pythonコードを実行して出力を確認する。"

    if False:
        # try4: thought
        print("try4: thought")
        input_dict = {
            "input": "please exec print('test output')",
        }
        brain.use_agent(AgentName.THOUGHT)
        result = brain.invoke(input_dict)
        print("result=", result)
        assert "test output" in result["output"]


if __name__ == "__main__":
    test()
