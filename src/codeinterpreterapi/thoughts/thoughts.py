from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from codeinterpreterapi.brain.params import CodeInterpreterParams
from codeinterpreterapi.llm.llm import prepare_test_llm
from codeinterpreterapi.thoughts.base import MyToTChain
from codeinterpreterapi.thoughts.checker import create_tot_chain_from_llm


class CodeInterpreterToT(RunnableSerializable):
    tot_chain: MyToTChain = None

    def __init__(self, llm=None, is_ja=True, is_simple=False):
        print("XXX llm=", llm)
        super().__init__()
        self.tot_chain = create_tot_chain_from_llm(llm=llm, is_ja=is_ja, is_simple=is_simple)

    def run(self, input: Input) -> Output:
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
    def get_runnable_tot_chain(cls, ci_params: CodeInterpreterParams, is_simple: bool = False) -> RunnableSerializable:
        # ToTChainのインスタンスを作成
        tot_chain = cls(llm=ci_params.llm_tools, is_ja=ci_params.is_ja, is_simple=is_simple)
        return tot_chain


def test():
    llm, llm_tools, runnable_config = prepare_test_llm()
    ci_params = CodeInterpreterParams.get_test_params(llm=llm, llm_tools=llm_tools, runnable_config=runnable_config)
    tot_chain = CodeInterpreterToT.get_runnable_tot_chain(ci_params=ci_params)
    tot_chain.invoke({"input": sample2})


sample1 = "pythonで円周率を表示するプログラムを実行してください。"
sample2 = """SVG画像を自動生成するプログラムの要件を以下のように定義します。

目的:

電子書籍のヘッダ画像を自動生成すること
別のコンテンツ生成プログラムが出力したSVGファイルを入力として受け取ること
入力SVGファイルを指定の要件に従って加工し、新たなSVGファイルとして出力すること
機能要件:

グリッドレイアウト機能の実装

指定したグリッドサイズ(行数、列数)に基づいて要素を配置できるようにする
グリッドの各セルに対して要素を割り当てられるようにする
グリッドのサイズや間隔を柔軟に設定できるようにする
SVG要素の配置と編集

グリッド上の指定した位置にSVG要素(テキスト、図形、画像など)を配置できるようにする
配置する要素の属性(サイズ、色、フォントなど)を柔軟に設定できるようにする
既存のSVG要素を削除、移動、変更できるようにする
外部画像ファイルの読み込みと配置

PNGやJPEGなどの外部画像ファイルを読み込んでSVGファイルに埋め込めるようにする
読み込んだ画像をグリッド上の指定した位置に配置できるようにする
画像のサイズを変更できるようにする
SVGファイルの入出力

SVGファイルを入力として読み込み、加工後のSVGファイルを出力できるようにする
出力ファイルのファイル名やパスを指定できるようにする
非機能要件:

Python3とsvgwriteライブラリを使用して実装すること
コードはモジュール化し、再利用性と保守性を高めること
エラーハンドリングを適切に行い、ログ出力を行うこと
コードにはコメントを付けて可読性を高めること
実装の進め方:

svgwriteを使ったSVGファイルの基本的な読み込み、編集、出力の機能を実装する
グリッドレイアウト機能を実装し、要素を配置できるようにする
外部画像ファイルの読み込みと配置機能を実装する
入力SVGファイルを読み込んで、指定の要件に従って加工し、新たなSVGファイルを出力する一連の処理を実装する
細かい仕様について検討し、機能を拡張する
テストを行い、不具合を修正する
ドキュメントを整備し、コードをリファクタリングする
まずはこの要件定義に基づいて、各機能の実装に着手してください。実装方法や詳細な手順は、要件に合わせて適宜ご判断ください。



作業フォルダは/app/workを使ってください。

全ての処理は自動で実施して結果とプログラムだけ報告してください。
"""

if __name__ == "__main__":
    test()
