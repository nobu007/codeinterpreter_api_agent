# LCEL version of.
# https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/tot/thought_generation.py
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableSequence, RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.prompts import get_cot_prompt, get_propose_prompt

from codeinterpreterapi.thoughts.tot import create_tot_chain_from_llm


class CodeInterpreterToT(RunnableSerializable):
    tot_chain: ToTChain = None

    def __init__(self, llm=None):
        super().__init__()
        self.tot_chain = create_tot_chain_from_llm(llm=llm)

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
    def get_runnable_tot_chain(
        cls,
        llm=None,
    ):
        # ToTChainのインスタンスを作成
        tot_chain = cls(
            llm=llm,
        )
        return tot_chain


class BaseThoughtGenerationStrategyRunnableSequence(RunnableSequence):
    """
    Base class for a thought generation strategy.
    """

    c: int = 3
    """The number of children thoughts to propose at each step."""

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        """
        Generate the next thought given the problem description and the thoughts
        generated so far.
        """
        return ""


class SampleCoTStrategyRunnableSequence(BaseThoughtGenerationStrategyRunnableSequence):
    """
    Sample strategy from a Chain-of-Thought (CoT) prompt.

    This strategy works better when the thought space is rich, such as when each
    thought is a paragraph. Independent and identically distributed samples
    lead to diversity, which helps to avoid repetition.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_cot_prompt)

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        response_text = self.predict_and_parse(
            problem_description=problem_description, thoughts=thoughts_path, **kwargs
        )
        return response_text if isinstance(response_text, str) else ""


class ProposePromptStrategyRunnableSequence(SampleCoTStrategyRunnableSequence):
    """
    Strategy that is sequentially using a "propose prompt".

    This strategy works better when the thought space is more constrained, such
    as when each thought is just a word or a line. Proposing different thoughts
    in the same prompt completion helps to avoid duplication.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt)
    tot_memory: Dict[Tuple[str, ...], List[str]] = Field(default_factory=dict)

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        if thoughts_path not in self.tot_memory or not self.tot_memory[thoughts_path]:
            new_thoughts = self.invoke(
                problem_description=problem_description,
                thoughts=thoughts_path,
                n=self.c,
                **kwargs,
            )
            if not new_thoughts:
                return ""
            if isinstance(new_thoughts, list):
                self.tot_memory[thoughts_path] = new_thoughts[::-1]
            else:
                return ""
        return self.tot_memory[thoughts_path].pop()


def test():
    tot_chain = CodeInterpreterToT.get_runnable_tot_chain()
    tot_chain.invoke({"input": "pythonで円周率を表示するプログラムを実行してください。"})


if __name__ == "__main__":
    test()
