from typing import Any, Dict

from langchain import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


def create_complement_input(prompt: PromptTemplate, remap: Dict = None) -> Runnable:
    """_summary_

    Args:
        prompt (PromptTemplate): _description_
        remap (Dict, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if remap is None:
        # key: original inputs key(ex: input), value: after replace key(ex: question)
        remap = {"input": "question"}

    def complement_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        complemented_dict = input_dict.copy()
        # step1: remap
        for key, value in remap.items():
            print("key, value =", key, value)
            if key in complemented_dict:
                complemented_dict[value] = complemented_dict[key]

        # step2: fill empty
        for key in prompt.input_variables:
            if key not in complemented_dict:
                complemented_dict[key] = ""

        print("complemented_dict=", complemented_dict)
        return complemented_dict

    return RunnableLambda(complement_input)
