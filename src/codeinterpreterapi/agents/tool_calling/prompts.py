from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_MESSAGE_TEMPLATE = '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            {tools}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
              "action": "Final Answer",
              "action_input": "Final response to human"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary.
            Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
            '''

HUMAN_MESSAGE_TEMPLATE = '''{input}

            {agent_scratchpad}

            (reminder to respond in a JSON blob no matter what)'''

SYSTEM_MESSAGE_TEMPLATE_JA = '''初期質問にできる限り丁寧かつ正確に答えてください。以下のツールが利用可能です:
    {tools}

1つの $JSON_BLOB では常に単一のアクションで応答してください。
action は (TOOL_NAME) 、action_input は (INPUT) を使ってください。
Valid "action" values: "Final Answer" or {tool_names}

フォーマットは次のようになります。

            ```
            {{
              "action": $TOOL_NAME,
              "action_input": $INPUT
            }}
            ```

その後のシーケンスは以下の形式に従ってください:

Question: 初期質問に正しく答えるための質問
Thought: 前後のステップを検討する
Action:

```
$JSON_BLOB
```

Observation: アクションの結果

(Thought/Action/Observation をN回繰り返す）

Thought: 何を答えるべきかわかった
Action:

```
{{
  "action": "最終回答",
  "action_input": "初期質問への最終的な回答"
}}
```

それでは始めましょう。常に単一のアクションの有効なJSONブロブで応答することを忘れないでください。
必要に応じてツールを使用してください。
適切な場合は直接回答してください。
            '''

HUMAN_MESSAGE_TEMPLATE_JA = '''{input}

            {agent_scratchpad}

            リマインダ： 何があっても $JSON_BLOB だけで応答するようにしてください。
'''


def create_tool_calling_agent_prompt(is_ja: bool = True):
    if is_ja:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE_JA),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", HUMAN_MESSAGE_TEMPLATE_JA),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE_TEMPLATE),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", HUMAN_MESSAGE_TEMPLATE),
            ]
        )
    return prompt
