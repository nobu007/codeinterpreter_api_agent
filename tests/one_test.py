from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.planners.planners import CodeInterpreterPlanner


def test():
    sample = "ステップバイステップで2*5+2を計算して。"
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    planner = CodeInterpreterPlanner.choose_planner(ci_params=ci_params)
    result = planner.invoke({"input": sample, "agent_scratchpad": "", "messages": [sample]})
    print("result=", result.content)


if __name__ == "__main__":
    test()
