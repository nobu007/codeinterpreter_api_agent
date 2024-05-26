from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_experimental.tot.base import ToTChain

from codeinterpreterapi.thoughts.tot import create_tot_chain_from_llm


class CodeInterpreterToT(RunnableSerializable):
    tot_chain: ToTChain = None

    def __init__(self, llm=None, is_ja=True, is_simple=False):
        super().__init__()
        self.tot_chain = create_tot_chain_from_llm(llm=llm, is_ja=is_ja, is_simple=is_simple)

    def run(self, input: Input):
        problem_description = input["input"]
        return self.tot_chain.run(problem_description=problem_description)

    def __call__(self, input: Input) -> Dict[str, str]:
        return self.run(input)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.run(input)

    def batch(self, inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
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

    @classmethod
    def get_runnable_tot_chain(cls, llm=None, is_ja=True, is_simple=False):
        # ToTChainのインスタンスを作成
        tot_chain = cls(llm=llm, is_ja=is_ja, is_simple=is_simple)
        return tot_chain


def test():
    tot_chain = CodeInterpreterToT.get_runnable_tot_chain()
    tot_chain.invoke({"input": "pythonで円周率を表示するプログラムを実行してください。"})


if __name__ == "__main__":
    test()
