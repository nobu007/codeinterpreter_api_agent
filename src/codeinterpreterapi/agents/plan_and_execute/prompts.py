from textwrap import dedent

from langchain_core.prompts import PromptTemplate
from langchain_experimental.tot.prompts import JSONListOutputParser

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""

TASK_PREFIX = """{objective}

"""

TOOLS_PREFIX = (
    """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
)
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
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
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""
SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Use tools if necessary. Respond directly if appropriate.
Format is like this.
Action:```$JSON_BLOB```
Observation:
Thought:"""

TOOLS_PREFIX_JA = """できる限り丁寧かつ正確に人間の質問に答えてください。以下のツールが利用可能です:"""

FORMAT_INSTRUCTIONS_JA = """
始めましょう！常に単一のアクションの有効なJSONブロブで応答することを忘れないでください。
必要に応じてツールを使用してください。適切な場合は直接回答してください。
フォーマットは次のようにしてください。
Action:```$JSON_BLOB```
Observation:
Thought:
"""
SUFFIX_JA = """それでは始めましょう。常に単一のアクションの有効なJSON_BLOBで応答してください。
必要に応じてツールを使用してください。適切な場合は直接回答してください。
フォーマットは次のようにしてください。
Action:```$JSON_BLOB```
Observation:
Thought:"""


def get_tools_prefix_prompt(is_ja: bool) -> str:
    """Get the prefix prompt for plan_and_execute."""
    if is_ja:
        return TOOLS_PREFIX_JA
    else:
        return TOOLS_PREFIX


def get_plan_and_execute_prompt(is_ja: bool) -> str:
    """Get the main prompt for plan_and_execute."""
    if is_ja:
        return FORMAT_INSTRUCTIONS_JA
    else:
        return FORMAT_INSTRUCTIONS
