from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_MESSAGE_TEMPLATE = '''
You are a supervisor tasked with managing a conversation between the following workers:
{members}.

Given the following user request, respond with the worker to act next.
Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
'''

SYSTEM_MESSAGE_TEMPLATE_JA = '''
あなたは次のワーカー間の会話を管理するスーパーバイザーです。
{members}

ユーザー リクエストが与えられると、
ワーカーに次のアクションを指示します。
各ワーカーはタスクを実行し、結果とステータスを応答します。
終了したら、FINISH で応答します。
'''


def create_supervisor_agent_prompt_next(is_ja: bool = True):
    if is_ja:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE_JA),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "上記の会話を踏まえて、次に誰が行動すべきでしょうか?"
                    " それとも、終了すべきでしょうか? 次のいずれかを選択してください: {options}",
                ),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        )
    return prompt


def create_supervisor_agent_prompt(is_ja: bool = True):
    if is_ja:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE_JA),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "上記の会話を踏まえて、次に誰が行動すべきでしょうか?"
                    " それとも、終了すべきでしょうか? 次のいずれかを選択してください: {options}",
                ),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        )
    return prompt
