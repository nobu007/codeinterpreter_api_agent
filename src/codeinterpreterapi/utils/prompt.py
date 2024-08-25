from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from codeinterpreterapi.brain.params import CodeInterpreterParams

COMMON_PROMPT_HEADER_JA = """
あなたは、あらゆる状況に対応できる能力を持った万能のagentです。
あなたが所属するシステムは全体で人間以上の能力を発揮できるAGIシステムです。
役割を正しく理解して他のagentと協調することに注意して作業してください。


"""
COMMON_PROMPT_ENVIRONMENT_JA = """
bashのあらゆるコマンドを使えます。
インターネットにアクセスできます。
新しいパッケージをpipでインストールできます。

## (重要)カレントディレクトリの注意
bashコマンド実行時のカレントディレクトリは"/app/work/"です。
コードが書かれたpythonファイル(*.py)もここに配置されます。
試験資材は"/app/work/inputs"にあります。
出力ファイルは"/app/work/outputs"に作ってください。
一方、pythonコード実行時のカレントディレクトリは"/app/codeinterpreter_api_agent/src/codeinterpreterapi/invoke_tasks"です。
正しく動作させるためにpythonプログラム内では必ず絶対パスで処理してください。
file関連でエラーが出た場合はカレントディレクトリをprintするなどで原因を確認してください。


"""


class PromptUpdater:
    @staticmethod
    def show_prompt(prompt: PromptTemplate):
        dummy_input = {var: "" for var in prompt.input_variables}
        prompt_str = prompt.format(**dummy_input)
        print(f"show_prompt: {prompt_str}")

    @staticmethod
    def show_chat_prompt(prompt: ChatPromptTemplate):
        print("show_chat_prompt:")
        for message in prompt.messages:
            if isinstance(message, SystemMessagePromptTemplate):
                print(f"System: {PromptUpdater.show_prompt(message.prompt)}")
            elif isinstance(message, HumanMessagePromptTemplate):
                print(f"Human: {message.prompt.template}")
            elif isinstance(message, AIMessagePromptTemplate):
                print(f"AI: {message.prompt.template}")
            elif isinstance(message, MessagesPlaceholder):
                print(f"MessagesPlaceholder: {message.variable_name}")
            else:
                print(f"Other: {message}")

    @staticmethod
    def update_prompt(prompt: PromptTemplate, ci_params: CodeInterpreterParams) -> PromptTemplate:
        # Check if the prompt has the required input variables
        missing_vars = {"agent_info"}.difference(prompt.input_variables + list(prompt.partial_variables))
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        # Partial the prompt with tools and tool_names
        prompt = prompt.partial(
            agent_info=", ".join([agent_def.get_agent_info() for agent_def in ci_params.agent_def_list]),
        )
        return prompt

    @staticmethod
    def update_chat_prompt(prompt: ChatPromptTemplate, ci_params: CodeInterpreterParams) -> ChatPromptTemplate:
        # Check if the prompt has the required input variables
        missing_vars = {"agent_info"}.difference(prompt.input_variables)
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        # Create a new ChatPromptTemplate with updated messages
        updated_messages = []
        for message in prompt.messages:
            if isinstance(message, SystemMessagePromptTemplate):
                updated_content = PromptUpdater.update_prompt(message.prompt, ci_params)
                updated_messages.append(SystemMessagePromptTemplate(prompt=updated_content))
            elif isinstance(message, (HumanMessagePromptTemplate, AIMessagePromptTemplate)):
                updated_messages.append(message)
            elif isinstance(message, MessagesPlaceholder):
                updated_messages.append(message)
            else:
                raise ValueError(f"Unexpected message type: {type(message)}")

        return ChatPromptTemplate(messages=updated_messages)

    @staticmethod
    def update_and_show_chat_prompt(prompt: ChatPromptTemplate, ci_params: CodeInterpreterParams) -> ChatPromptTemplate:
        updated_prompt = PromptUpdater.update_chat_prompt(prompt, ci_params)
        PromptUpdater.show_chat_prompt(updated_prompt)
        return updated_prompt
