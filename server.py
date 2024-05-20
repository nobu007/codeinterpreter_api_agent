from gui_agent_loop_core.connector_impl.core_to_agent.connector_impl_codeinterpreter_api import (
    ConnectorImplCodeinterpreterApi,
)
from gui_agent_loop_core.gui_agent_loop_core import GuiAgentLoopCore

from codeinterpreterapi import CodeInterpreterSession


class CodeInterpreter(ConnectorImplCodeinterpreterApi):
    def __init__(self):
        model = "claude-3-haiku-20240307"
        # model = "gemini-1.5-pro-latest"
        # model = "gemini-1.0-pro"
        self.session = CodeInterpreterSession(model=model, verbose=True)
        self.status = self.session.start_local()

    # def chat_core(
    #     self,
    #     message: GuiAgentInterpreterChatMessages,
    #     display: bool = False,
    #     stream: bool = False,
    #     blocking: bool = False,
    # ) -> GuiAgentInterpreterChatResponseAny:
    #     try:
    #         message = """SVG画像を自動生成するプログラムの要件を以下のように定義します。

    #     目的:

    #     電子書籍のヘッダ画像を自動生成すること
    #     別のコンテンツ生成プログラムが出力したSVGファイルを入力として受け取ること
    #     入力SVGファイルを指定の要件に従って加工し、新たなSVGファイルとして出力すること
    #     機能要件:

    #     グリッドレイアウト機能の実装

    #     指定したグリッドサイズ(行数、列数)に基づいて要素を配置できるようにする
    #     グリッドの各セルに対して要素を割り当てられるようにする
    #     グリッドのサイズや間隔を柔軟に設定できるようにする
    #     SVG要素の配置と編集

    #     グリッド上の指定した位置にSVG要素(テキスト、図形、画像など)を配置できるようにする
    #     配置する要素の属性(サイズ、色、フォントなど)を柔軟に設定できるようにする
    #     既存のSVG要素を削除、移動、変更できるようにする
    #     外部画像ファイルの読み込みと配置

    #     PNGやJPEGなどの外部画像ファイルを読み込んでSVGファイルに埋め込めるようにする
    #     読み込んだ画像をグリッド上の指定した位置に配置できるようにする
    #     画像のサイズを変更できるようにする
    #     SVGファイルの入出力

    #     SVGファイルを入力として読み込み、加工後のSVGファイルを出力できるようにする
    #     出力ファイルのファイル名やパスを指定できるようにする
    #     非機能要件:

    #     Python3とsvgwriteライブラリを使用して実装すること
    #     コードはモジュール化し、再利用性と保守性を高めること
    #     エラーハンドリングを適切に行い、ログ出力を行うこと
    #     コードにはコメントを付けて可読性を高めること
    #     実装の進め方:

    #     svgwriteを使ったSVGファイルの基本的な読み込み、編集、出力の機能を実装する
    #     グリッドレイアウト機能を実装し、要素を配置できるようにする
    #     外部画像ファイルの読み込みと配置機能を実装する
    #     入力SVGファイルを読み込んで、指定の要件に従って加工し、新たなSVGファイルを出力する一連の処理を実装する
    #     細かい仕様について検討し、機能を拡張する
    #     テストを行い、不具合を修正する
    #     ドキュメントを整備し、コードをリファクタリングする
    #     まずはこの要件定義に基づいて、各機能の実装に着手してください。実装方法や詳細な手順は、要件に合わせて適宜ご判断ください。

    #     作業フォルダは/app/workを使ってください。

    #     全ての処理は自動で実施して結果とプログラムだけ報告してください。

    #         """

    #         is_stream = False
    #         if is_stream:
    #             # ChainExecutorのエラーが出て動かない
    #             """
    #             process_messages_gradio response_chunks= <async_generator object process_and_format_message at 0x7f9f2652bec0>
    #             llm stream start
    #             server chat chunk_response= type=<Type.MESSAGE: 'message'> role=<Role.USER: 'user'> content="Error in CodeInterpreterSession: AttributeError  - 'ChainExecutor' object has no attribute 'stream'" format='' code='' start=False end=False
    #             process_messages_gradio response= type=<Type.MESSAGE: 'message'> role=<Role.USER: 'user'> content="Error in CodeInterpreterSession: AttributeError  - 'ChainExecutor' object has no attribute 'stream'" format='' code='' start=False end=False
    #             memory.save_context full_response= Error in CodeInterpr
    #             """
    #             for chunk_str in self.session.generate_response_stream(message):
    #                 chunk_response = GuiAgentInterpreterChatResponse()
    #                 chunk_response.content = chunk_str
    #                 print("server chat chunk_response=", chunk_response)
    #                 yield chunk_response
    #         else:
    #             response = self.session.generate_response(message)
    #             print("server chat response(no stream)=", response)
    #             yield response

    #     except Exception as e:
    #         print(e)
    #         traceback.print_exc()
    #         error_response = {"role": GuiAgentInterpreterChatMessage.Role.ASSISTANT, "content": str(e)}
    #         yield error_response


interpreter = CodeInterpreter()
core = GuiAgentLoopCore()
core.launch_server(interpreter)
