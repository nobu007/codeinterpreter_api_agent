from typing import Sequence

from langchain.agents import AgentExecutor
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import MessageGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from codeinterpreterapi.agents.agents import CodeInterpreterAgent
from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.graphs.agent_wrapper_tool import AgentWrapperTool
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner
from codeinterpreterapi.supervisors.supervisors import CodeInterpreterSupervisor
from codeinterpreterapi.test_prompts.test_prompt import TestPrompt


@tool
def divide(a: float, b: float) -> int:
    """Return a / b."""
    return a / b


def create_agent_nodes(ci_params: CodeInterpreterParams):
    agent_nodes = []
    for agent_def in ci_params.agent_def_list:
        agent_executor = agent_def.agent_executor
        agent_node = create_agent_node(agent_executor)
        agent_nodes.append(agent_node)
    return agent_nodes


def create_agent_node(agent_executor: AgentExecutor):
    def agent_function(state, context):
        """
        AgentExecutorを実行し、結果を状態に追加する関数。

        Args:
            state (dict): 現在の状態。messagesキーを含む必要がある。
            context (dict): 実行コンテキスト。

        Returns:
            dict: 更新された状態。
        """
        # stateから必要な情報を取得
        messages = state["messages"]

        # AgentExecutorを実行
        result = agent_executor.invoke({"input": messages[-1].content})

        # 結果を新しいメッセージとして追加
        new_message = AIMessage(content=result["output"])
        state["messages"] = messages + [new_message]
        print(state)
        return state

    return ToolNode([agent_function])


def create_tool_node(
    state_graph: StateGraph,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
) -> StateGraph:
    state_graph.add_node("tools", ToolNode(tools))
    state_graph.add_node("chatbot", llm.bind_tools(tools))
    state_graph.add_edge("tools", "chatbot")
    state_graph.add_conditional_edges("chatbot", tools_condition)
    return state_graph


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    _ = CodeInterpreterAgent.choose_agent_executors(ci_params=ci_params)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    _ = CodeInterpreterSupervisor.choose_supervisor(planner=planner, ci_params=ci_params)

    state_graph = MessageGraph()
    tools = AgentWrapperTool.create_agent_wrapper_tools(ci_params)
    # state_graph = create_tool_node(state_graph, llm_tools, tools)
    state_graph = create_agent_node(state_graph, llm_tools, tools)
    state_graph.set_entry_point("chatbot")
    compiled_graph = state_graph.compile()
    result = compiled_graph.invoke([("user", TestPrompt.svg_input_str)])
    print(result[-1].content)


if __name__ == "__main__":
    test()
