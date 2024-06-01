from typing import Any, Dict, Union

from langchain.prompts.base import BasePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence


def create_complement_input(prompt: Union[BasePromptTemplate, RunnableSequence], remap: Dict = None):
    """
    入力辞書のキーをリマップし、欠損しているキーを空文字列で補完する関数を作成します。

    Args:
        prompt (Union[BasePromptTemplate, Chain]): 使用するプロンプトテンプレートまたはチェーン。
        remap (Dict, optional): キーのリマップ辞書。キー：元の入力キー（例：input）、値：置換後のキー（例：question）。
                                デフォルトは None で、{"input": "question"} が使用されます。

    Returns:
        complement_input (Callable[[Dict[str, Any]], Dict[str, Any]]): 入力辞書を補完する関数。
    """
    if remap is None:
        # デフォルトのリマップ辞書
        # key: original inputs key(ex: input), value: after replace key(ex: question)
        remap = {"input": "question"}

    def complement_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        入力辞書を補完する関数。

        Args:
            input_dict (Dict[str, Any]): 補完対象の入力辞書。

        Returns:
            complemented_dict (Dict[str, Any]): 補完後の入力辞書。
        """
        complemented_dict = input_dict.copy()

        # ステップ1: キーのリマップ
        for key, value in remap.items():
            print("key, value =", key, value)
            if key in complemented_dict:
                complemented_dict[value] = complemented_dict[key]

        # ステップ2: プロンプトの入力変数に基づいて、欠損しているキーを空文字列で補完
        if isinstance(prompt, BasePromptTemplate):
            input_variables = prompt.input_variables
        elif isinstance(prompt, RunnableSequence):
            input_variables = prompt.first.input_variables
        else:
            raise ValueError("prompt must be either BasePromptTemplate or Chain")

        for key in input_variables:
            if key not in complemented_dict:
                complemented_dict[key] = ""

        print("complemented_dict=", complemented_dict)
        return complemented_dict

    return RunnableLambda(complement_input)
