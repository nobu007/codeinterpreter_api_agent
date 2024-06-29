import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor


class CodeInterpreterStateGraph:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.initialize_agent_info()
        self.graph = self.initialize_graph()
        self.app = self.graph.compile()
        self.node_descriptions_dict = {}
        self.node_agent_dict = {}

    def initialize_agent_info(self) -> None:
        # 各エージェントに対してノードを作成
        for agent_def in self.ci_params.agent_def_list:
            agent = agent_def.agent
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
        messages: Annotated[Sequence[BaseMessage], operator.add]  # メッセージの履歴

    def initialize_graph(self) -> StateGraph:
        workflow = StateGraph(CodeInterpreterStateGraph.GraphState)

        SUPERVISOR_AGENT_NAME = "supervisor_agent"
        workflow.add_node(SUPERVISOR_AGENT_NAME, self.ci_params.supervisor_agent)
        for agent_name, agent in self.node_agent_dict.items():
            workflow.add_node(agent_name, agent)
            workflow.add_edge(agent_name, SUPERVISOR_AGENT_NAME)

        # The supervisor populates the "next" field in the graph state
        # which routes to a node or finishes
        conditional_map = {k: k for k, _ in self.node_descriptions_dict}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges(SUPERVISOR_AGENT_NAME, lambda x: x["next"], conditional_map)
        # Finally, add entrypoint
        workflow.set_entry_point(SUPERVISOR_AGENT_NAME)

        graph = workflow.compile()

        return graph

    def run(self, input_data):
        # グラフを実行
        final_state = self.app.invoke(input_data)
        return final_state


def test():
    llm, llm_tools = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)

    sg = CodeInterpreterStateGraph(ci_params)
    test_input = "pythonで円周率を表示するプログラムを実行してください。"
    output = sg.invoke({"messages": [test_input]})
    print("output=", output)


if __name__ == "__main__":
    test()
