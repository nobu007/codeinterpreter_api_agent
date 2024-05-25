import os
from typing import Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory, HumanMessage
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity

sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()
print(problem_description)

#######
# The following code implements an LLM-based checker
#######


class MyChecker(ToTChecker):
    llm: Optional[BaseMemory] = None
    prompt: PromptTemplate = PromptTemplate(
        input_variables=["problem_description", "thoughts"],
        template="""
        Given the following problem description and a series of thoughts, evaluate the validity of the last thought.

        Problem Description:
        {problem_description}

        Thoughts:
        {thoughts}

        Evaluate the last thought and return one of the following:
        - VALID_FINAL if the last thought is a valid final solution to the problem.
        - VALID_INTERMEDIATE if the last thought is a valid intermediate step towards the solution.
        - INVALID if the last thought is invalid or contradicts the problem description.

        Evaluation:
        """,
    )

    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        prompt = self.prompt.format(problem_description=problem_description, thoughts="\n".join(thoughts))
        message = HumanMessage(content=prompt)
        evaluation = self.llm([message]).content.strip().upper()

        print("evaluation=", evaluation)

        if evaluation == "VALID_FINAL":
            return ThoughtValidity.VALID_FINAL
        elif evaluation == "VALID_INTERMEDIATE":
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID


#######
# Testing the MyChecker class above:
#######
def test_checker(tot_chain):
    checker = tot_chain.checker
    assert (
        checker.evaluate(problem_description, ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",))
        == ThoughtValidity.VALID_INTERMEDIATE
    )
    assert checker.evaluate(problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",)) == ThoughtValidity.VALID_FINAL
    assert (
        checker.evaluate(problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",))
        == ThoughtValidity.VALID_INTERMEDIATE
    )
    assert checker.evaluate(problem_description, ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",)) == ThoughtValidity.INVALID


#######
# Initialize and run the ToT chain,
# with maximum number of interactions k set to 30 and
# the maximum number of child thoughts c set to 8.
#######


def create(llm):
    checker = MyChecker()
    checker.llm = llm
    tot_chain = ToTChain.from_llm(llm=llm, checker=checker, k=30, c=5, verbose=True, verbose_llm=False)
    tot_chain.run(problem_description=problem_description)
    return tot_chain


def test_create():
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    model = "gemini-1.5-flash-latest"
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_output_tokens=1024,
    )
    create(llm)


if __name__ == "__main__":
    test_create()
