import json
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser

from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.prompts import determine_modifications_prompt


def get_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 4,
) -> Optional[List[str]]:
    if retry < 1:
        return None

    try:
        result_seq = determine_modifications_prompt | llm | JsonOutputParser()
        result = result_seq.invoke({"code": code})
        print("result=", result)
    except json.JSONDecodeError:
        result = ""
    if not result or not isinstance(result, dict) or "modifications" not in result:
        return get_file_modifications(code, llm, retry=retry - 1)
    return result["modifications"]


async def aget_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 4,
) -> Optional[List[str]]:
    if retry < 1:
        return None

    try:
        result_seq = determine_modifications_prompt | llm | JsonOutputParser()
        result = result_seq.invoke({"code": code})
        print("result=", result)
    except json.JSONDecodeError:
        result = ""
    if not result or not isinstance(result, dict) or "modifications" not in result:
        return await aget_file_modifications(code, llm, retry=retry - 1)
    return result["modifications"]


async def test() -> None:
    llm, llm_tools, runnable_config = prepare_test_llm()

    code = """
        import matplotlib.pyplot as plt

        x = list(range(1, 11))
        y = [29, 39, 23, 32, 4, 43, 43, 23, 43, 77]

        plt.plot(x, y, marker='o')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Data Plot')

        plt.show()
        """

    result = get_file_modifications(code, llm)
    assert result == []  # This is no change for the file system.


if __name__ == "__main__":
    import asyncio

    import dotenv

    dotenv.load_dotenv()

    asyncio.run(test())
