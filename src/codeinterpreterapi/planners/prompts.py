from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

SYSTEM_MESSAGE_TEMPLATE = '''
    You are a super agent manager.
    First of all, understand the problem.
    Please make a plan to solve it.

    You cau use AI agent here.
    {agent_info}.

    Constraints:.
    - Think step-by-step and be clear about the actions to be taken at each step.
    - If the required information is missing, answer with agent_name=None and ask the user in task_description.
    - If the required agent for execution does not exist, answer with agent_name=None and answer about the required agent in the task_description.
    - Once sufficient information is obtained to complete the task, the final work plan should be output.
    - The last step should be agent_name=<END_OF_PLAN>.

Translated with DeepL.com (free version)
'''

SYSTEM_MESSAGE_TEMPLATE_JA = '''
    あなたは優秀なAIエージェントを管理するシニアエンジニアです。
    まず、問題を理解し、問題を解決するための計画を立てましょう。

    利用可能なAI agentのリスト:
    {agent_info}

    制約条件:
    - 段階的に考え、各ステップで取るアクションを明確にすること。
    - 必要な情報が不足している場合は、agent_name=Noneで回答し、task_description でユーザーに質問すること。
    - 実行に必要なagentが存在しない場合は、agent_name=Noneで回答し、task_description で必要なagentについて回答すること。
    - タスクを完了するために十分な情報が得られたら、最終的な作業計画を出力すること。
    - 最後のステップはagent_name=<END_OF_PLAN>とすること。
    - 各ステップの思考と出力は日本語とする。

'''


def create_planner_agent_prompt(is_ja: bool = True) -> PromptTemplate:
    if is_ja:
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "agent_info"], template=SYSTEM_MESSAGE_TEMPLATE_JA
        )
    else:
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "agent_info"], template=SYSTEM_MESSAGE_TEMPLATE
        )
    return prompt


def create_planner_agent_chat_prompt(is_ja: bool = True) -> ChatPromptTemplate:
    if is_ja:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE_JA),
                MessagesPlaceholder(variable_name="messages"),
                ("user", "上記の会話を踏まえて、最終的な作業計画を日本語で出力してください。"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
                ("user", "Given the conversation above, please output final plan."),
            ]
        )
    return prompt
