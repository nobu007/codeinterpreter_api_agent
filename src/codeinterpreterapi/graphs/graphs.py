import functools
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import END, StateGraph

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.callbacks.util import show_callback_info
from codeinterpreterapi.graphs.tool_node.tool_node import create_agent_nodes
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


def agent_node(state, agent, name):
    print(f"agent_node {name} node!")
    print("  state keys=", state.keys())
    inputs = state
    if "input" not in inputs:
        # inputs["input"] = state["question"]
        inputs["input"] = str(state["messages"])
    # inputs["agent_scratchpad"] = str(state["messages"])
    result = agent.invoke(inputs)
    print("agent_node type(result)=", type(result))
    if "output" in result:
        state["messages"].append(str(result["output"]))
    return state


def supervisor_node(state, supervisor, name):
    print(f"supervisor_node {name} node!")
    print("  state keys=", state.keys())
    result = supervisor.invoke(state)
    print("supervisor_node type(result)=", type(result))
    # print("supervisor_node result=", result)

    state["question"] = state["messages"][0]
    if result is None:
        state["next"] = "FINISH"
    elif isinstance(result, dict):
        print("supervisor_node type(result)=", type(result))
        # if "output" in result:
        #     state["messages"].append(str(result["output"]))
        if "next" in result:
            state["next"] = result["next"]
            print("supervisor_node result(dict) next=", result["next"])
        state["messages"].append(f"次のagentは「{result.next}」です。")
    elif hasattr(result, "next"):
        # RouteSchema object
        state["next"] = result.next
        state["messages"].append(f"次のagentは「{result.next}」です。")
        print("supervisor_node result(RouteSchema) next=", result.next)
    else:
        state["next"] = "FINISH"

    if state["next"] == "FINISH":
        state["messages"].append("処理完了です。")
    else:
        next_agent = state["next"]
        state["messages"].append(f"次のagentは「{next_agent}」です。")

    return state


# グラフで使用する変数(状態)を定義
class GraphState(TypedDict):
    # llm_bind_tool: BaseLLM  # ツールが紐付けされたllmモデル
    # emb_model: HuggingFaceEmbeddings  # Embeddingsモデル
    question: str  # 質問文
    # documents: List[Document]  # indexから取得したドキュメントのリスト
    messages: Annotated[Sequence[BaseMessage], operator.add] = []
    # intermediate_steps: str = ""


def should_end(state: GraphState):
    last_message = state["messages"][-1]
    return "FINAL ANSWER:" in last_message.content


class CodeInterpreterStateGraph:
    def __init__(self, ci_params: CodeInterpreterParams):
        self.ci_params = ci_params
        self.node_descriptions_dict = {}
        self.node_agent_dict = {}
        # self.initialize_agent_info()
        self.graph = self.initialize_graph()

    # メッセージ変更関数の準備
    def _modify_messages(self, messages: list[AnyMessage]):
        show_callback_info("_modify_messages=", "messages", messages)
        last_message = messages[0]
        return [last_message]

    def initialize_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        agent_nodes = create_agent_nodes(self.ci_params)
        is_first = True
        for i, agent_node in enumerate(agent_nodes):
            agent_name = f"agent{i}"
            workflow.add_node(agent_name, agent_node)
            # エージェントの実行後、即座に終了
            workflow.add_edge(agent_name, END)
            if is_first:
                workflow.set_entry_point(agent_name)
                is_first = False
            break

        return workflow.compile()

    def initialize_graph2(self) -> StateGraph:
        workflow = StateGraph(GraphState)
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
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor(planner=planner, ci_params=ci_params)

    sg = CodeInterpreterStateGraph(ci_params=ci_params)
    output = sg.run({"input": TestPrompt.svg_input_str, "messages": [TestPrompt.svg_input_str]})
    print("output=", output)


if __name__ == "__main__":
    test()
