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
'''

SYSTEM_MESSAGE_TEMPLATE_JA = '''
    あなたは優秀なAIエージェントを管理するシニアエンジニアです。
    次の手順で問題を解決するための計画(CodeInterpreterPlanList)を作成してください。

    # 手順
    手順１： 問題を理解する
    手順２： 利用可能なAI agentのリスト(agent_info)を確認する
    手順３： 問題解決に利用するべきAI agentをピックアップする
    手順４： ピックアップしたAI agentを利用する順番を決定する
    手順５： 各AI agentに与えるinput/outputについて検討する
    手順６： CodeInterpreterPlanList として最終的な計画を出力する

    # 利用可能なAI agent
    {agent_info}

    # 制約条件
    - ステップバイステップで精密に思考し回答する。
    - 作業として何を求められているか正しく理解する。
    - AI agentの機能を正確に理解してから回答する。
    - 各ステップの入力と出力を明確にする。
    - agent_infoに示されたagent_name以外のagentを利用しない。
    - 次の場合は計画を作成せずに長さ0のリストを返す。
      -- 利用可能なagentが不足している
      -- 簡単な問題のため計画が不要
      -- 難しすぎる問題で計画を実行しても有意義な結果が得られない
    - 各ステップの思考と出力は日本語とする。

    # 問題
'''


def create_planner_agent_prompt(is_ja: bool = True) -> PromptTemplate:
    if is_ja:
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "agent_info"],
            template=SYSTEM_MESSAGE_TEMPLATE_JA + "\n{input}",
        )
    else:
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "agent_info"], template=SYSTEM_MESSAGE_TEMPLATE + "\n{input}"
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
