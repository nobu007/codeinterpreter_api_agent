import os
from typing import Optional, Tuple

import spacy
from codeinterpreterapi.thoughts.thought_generation import (
    MyProposePromptStrategy,
    MyProposePromptStrategyJa,
    MySampleCoTStrategy,
    MySampleCoTStrategyJa,
)
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from spacy import Language

sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
sudoku_problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- |で区切られた4つの数字は、first row, second row, third row, 4th row をそれぞれ表します。
- 各rowとcolumnの4つの数字は重複していけません。例えば1,2,3,4や1,2,4,3はOKですが、1,2,2,3はNGです。
- さらに、2x2 のサブグリッド(全4個)でも4つの数字は重複してはいけません。
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()

#######
# The following code implements an LLM-based checker
#######


class MyToTChecker(ToTChecker):
    llm: Optional[BaseMemory] = None
    prompt: PromptTemplate = PromptTemplate(
        input_variables=["problem_description", "thoughts"],
        template="""
        次の Problem Description に示す問題を解決するための thoughts について、
        [VALID_FINAL|VALID_INTERMEDIATE|INVALID]のどれか１つだけを選んで１行目に回答してください。
        そして、選択した理由を2行目以降でできるだけ詳しく説明してください。

        Problem Description: 解決するべき問題です。
        {problem_description}

        Thoughts: 解決の手続きについての思考です。
        {thoughts}

        正しい回答を選ぶために下記のガイドラインに従ってください。
        - VALID_FINAL: 最終的な thoughts が問題の解決に最適であると確認したときに選びます。
        - VALID_INTERMEDIATE: 最終的な thoughts が問題の解決に適しているが、解決には思考をさらに進める必要があるときに選びます。
        - INVALID: 最終的な thoughts が問題を解決できないとき、内容がルールに違反していたとき、明らな間違いがあり思考をやり直す必要があるときに選びます。

        Evaluation:
        """,
    )
    nlp: Language = spacy.load("en_core_web_md")

    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        print("thoughts=", thoughts)
        evaluation = self.prompt | self.llm | StrOutputParser()
        llm_output = evaluation.invoke({"problem_description": problem_description, "thoughts": thoughts})

        print("llm_output=", llm_output)
        final_judge = self.judge_llm_output(llm_output)
        print("final_judge=", final_judge)
        return final_judge

    def judge_llm_output(self, llm_output) -> ThoughtValidity:
        llm_output_1st_line = llm_output.split("\n")[0]
        thought_validity_candidates = ["VALID_FINAL", "VALID_INTERMEDIATE", "INVALID"]
        for thought_validity in thought_validity_candidates:
            if thought_validity in llm_output_1st_line:
                return self.get_thought_validity(thought_validity)

        # nlp judge
        actual = self.nlp(llm_output_1st_line)
        options_nlp = ["FINAL", "INTERMEDIATE", "INVALID"]
        similarities = [actual.similarity(self.nlp(option)) for option in options_nlp]
        print("similarities=", similarities)
        best_match_index = similarities.index(max(similarities))
        best_match = thought_validity_candidates[best_match_index]

        print(f"Best match: {best_match} with similarity {similarities[best_match_index]}")
        return self.get_thought_validity(best_match)

    def get_thought_validity(self, thought_validity) -> ThoughtValidity:
        if thought_validity == "VALID_FINAL":
            return ThoughtValidity.VALID_FINAL
        elif thought_validity == "VALID_INTERMEDIATE":
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID


#######
# Testing the MyChecker class above:
#######
def test_checker():
    tot_chain = create_tot_chain_from_llm(prepare_test_llm())
    checker = tot_chain.checker
    assert (
        checker.evaluate(sudoku_problem_description, ("3,*,1,2|1,*,3,*|*,1,*,3|4,*,*,1",))
        == ThoughtValidity.VALID_INTERMEDIATE
    )
    assert (
        checker.evaluate(sudoku_problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",))
        == ThoughtValidity.VALID_FINAL
    )
    assert (
        checker.evaluate(sudoku_problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",))
        == ThoughtValidity.VALID_INTERMEDIATE
    )
    assert checker.evaluate(sudoku_problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,2,3,1",)) == ThoughtValidity.INVALID


#######
# Initialize and run the ToT chain,
# with maximum number of interactions k set to 30 and
# the maximum number of child thoughts c set to 8.
#######


def create_tot_chain_from_llm(llm=None, is_ja=True, is_simple=False):
    checker = MyToTChecker()
    if llm is None:
        llm = prepare_test_llm()
    checker.llm = llm
    if is_ja:
        if is_simple:
            tot_strategy_class = MySampleCoTStrategyJa
        else:
            tot_strategy_class = MyProposePromptStrategyJa
    else:
        if is_simple:
            tot_strategy_class = MySampleCoTStrategy
        else:
            tot_strategy_class = MyProposePromptStrategy

    tot_chain = ToTChain.from_llm(
        llm=llm,
        checker=checker,
        k=20,
        c=3,
        verbose=True,
        tot_strategy_class=tot_strategy_class,
        verbose_llm=False,
    )
    return tot_chain


def prepare_test_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    model = "gemini-1.5-flash-latest"
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_output_tokens=1024 * 4,
    )
    return llm


def test_create():
    tot_chain = create_tot_chain_from_llm(llm=prepare_test_llm(), is_simple=True)
    tot_chain.run(problem_description=sudoku_problem_description)


if __name__ == "__main__":
    test_checker()
    test_create()
