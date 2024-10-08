# LCEL version of.
# https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/tot/thought_generation.py
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import BasePromptTemplate
from pydantic import Field
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from langchain_experimental.tot.thought_generation import ProposePromptStrategy, SampleCoTStrategy

from codeinterpreterapi.thoughts.prompts import (
    get_cot_prompt,
    get_cot_prompt_ja,
    get_propose_prompt,
    get_propose_prompt_ja,
)


class MySampleCoTStrategy(SampleCoTStrategy):
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


class MySampleCoTStrategyJa(SampleCoTStrategy):
    """
    Sample strategy from a Chain-of-Thought (CoT) prompt.

    This strategy works better when the thought space is rich, such as when each
    thought is a paragraph. Independent and identically distributed samples
    lead to diversity, which helps to avoid repetition.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_cot_prompt_ja)

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


class MyProposePromptStrategy(ProposePromptStrategy):
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
            new_thoughts = self.predict_and_parse(
                problem_description=problem_description,
                thoughts=thoughts_path,
                n=self.c,
                **kwargs,
            )
            print("new_thoughts=", new_thoughts)
            if not new_thoughts:
                return ""
            if isinstance(new_thoughts, list):
                self.tot_memory[thoughts_path] = new_thoughts[::-1]
            else:
                return ""
        return self.tot_memory[thoughts_path].pop()


class MyProposePromptStrategyJa(ProposePromptStrategy):
    """
    Strategy that is sequentially using a "propose prompt".

    This strategy works better when the thought space is more constrained, such
    as when each thought is just a word or a line. Proposing different thoughts
    in the same prompt completion helps to avoid duplication.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt_ja)
    tot_memory: Dict[Tuple[str, ...], List[str]] = Field(default_factory=dict)
    tot_checker: ToTChecker = None  # Checkerを外部から渡すようにする

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        tot_checker: ToTChecker = None,
        **kwargs: Any,
    ) -> str:
        self.tot_checker = tot_checker
        if thoughts_path not in self.tot_memory or not self.tot_memory[thoughts_path]:
            new_thoughts = self.predict_and_parse(
                problem_description=problem_description,
                thoughts=thoughts_path,
                n=self.c,
                **kwargs,
            )
            print("thoughts_path=", thoughts_path)
            print("new_thoughts=", new_thoughts)
            if not new_thoughts:
                return ""
            if isinstance(new_thoughts, list):
                self.tot_memory[thoughts_path] = new_thoughts[::-1]
            else:
                self.tot_memory[thoughts_path] = [new_thoughts]

        return self.evaluate_and_pop_thought(problem_description, thoughts_path)

    def evaluate_and_pop_thought(self, problem_description: str, thoughts_path: Tuple[str, ...]) -> str:
        while self.tot_memory[thoughts_path]:
            thought = self.tot_memory[thoughts_path].pop()
            validity = self.tot_checker.evaluate(problem_description, thoughts_path + (thought,))

            if validity == ThoughtValidity.VALID_FINAL:
                return thought
            elif validity == ThoughtValidity.VALID_INTERMEDIATE:
                return thought
            else:
                continue

        return ""
