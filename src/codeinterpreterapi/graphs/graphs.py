import functools
import operator
from typing import Annotated, Sequence, TypedDict

from langchain.callbacks import StdOutCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


def agent_node(state, agent, name):
    print(f"agent_node {name} node!")
    print("  state=", state)
    if "input" not in state:
        state["input"] = state["question"]
    result = agent.invoke(state)
    print("agent_node result=", result)
    if "output" in result:
        state["messages"].append(str(result["output"]))
    return state


def supervisor_node(state, supervisor, name):
    print(f"supervisor_node {name} node!")
    print("  state=", state)
    result = supervisor.invoke(state)
    print("supervisor_node type(result)=", type(result))
    print("supervisor_node result=", result)

    if result is None:
        state["next"] = "FINISH"
    elif isinstance(result, dict):
        # if "output" in result:
        #     state["messages"].append(str(result["output"]))
        if "next" in result:
            state["next"] = result.next
        state["messages"].append(f"次のagentは「{result.next}」です。")
    elif hasattr(result, "next"):
        # RouteSchema object
        state["next"] = result.next
        state["messages"].append(f"次のagentは「{result.next}」です。")
    else:
        state["next"] = "FINISH"

    if state["next"] == "FINISH":
        state["messages"].append("処理完了です。")
    else:
        next_agent = state["next"]
        state["messages"].append(f"次のagentは「{next_agent}」です。")

    return state


class CodeInterpreterStateGraph:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.node_descriptions_dict = {}
        self.node_agent_dict = {}
        self.initialize_agent_info()
        self.graph = self.initialize_graph()

    def initialize_agent_info(self) -> None:
        # 各エージェントに対してノードを作成
        for agent_def in self.ci_params.agent_def_list:
            agent = agent_def.agent_executor
            agent_name = agent_def.agent_name
            agent_role = agent_def.agent_role

            self.node_descriptions_dict[agent_name] = agent_role
            self.node_agent_dict[agent_name] = agent

    # グラフで使用する変数(状態)を定義
    class GraphState(TypedDict):
        # llm_bind_tool: BaseLLM  # ツールが紐付けされたllmモデル
        # emb_model: HuggingFaceEmbeddings  # Embeddingsモデル
        question: str  # 質問文
        # documents: List[Document]  # indexから取得したドキュメントのリスト
        messages: Annotated[Sequence[BaseMessage], operator.add] = []
        # intermediate_steps: str = ""

    def initialize_graph(self) -> StateGraph:
        workflow = StateGraph(CodeInterpreterStateGraph.GraphState)

        SUPERVISOR_AGENT_NAME = "supervisor_agent"
        supervisor_node_replaced = functools.partial(
            supervisor_node, supervisor=self.ci_params.supervisor_agent, name=SUPERVISOR_AGENT_NAME
        )
        workflow.add_node(SUPERVISOR_AGENT_NAME, supervisor_node_replaced)
        for agent_name, agent in self.node_agent_dict.items():
            agent_node_replaced = functools.partial(agent_node, agent=agent, name=agent_name)
            workflow.add_node(agent_name, agent_node_replaced)
            workflow.add_edge(agent_name, SUPERVISOR_AGENT_NAME)

        # The supervisor populates the "next" field in the graph state
        # which routes to a node or finishes
        conditional_map = {k: k for k, _ in self.node_descriptions_dict.items()}
        conditional_map["FINISH"] = END
        print("conditional_map=", conditional_map)
        workflow.add_conditional_edges(SUPERVISOR_AGENT_NAME, lambda x: x["next"], conditional_map)
        # Finally, add entrypoint
        workflow.set_entry_point(SUPERVISOR_AGENT_NAME)

        graph = workflow.compile()

        return graph

    def run(self, input_data):
        # グラフを実行
        final_state = self.graph.invoke(input_data)
        return final_state


def test():
    llm, llm_tools = prepare_test_llm()
    config = RunnableConfig({'callbacks': [StdOutCallbackHandler()]})
    llm = llm.with_config(config)
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)

    sg = CodeInterpreterStateGraph(ci_params)
    output = sg.run({"input": TestPrompt.svg_input_str, "messages": [TestPrompt.svg_input_str]})
    print("output=", output)


if __name__ == "__main__":
    test()
