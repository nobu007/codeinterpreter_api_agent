from textwrap import dedent

from langchain_core.prompts import PromptTemplate
from langchain_experimental.tot.prompts import JSONListOutputParser


def get_cot_prompt() -> PromptTemplate:
    """Get the prompt for the Chain of Thought (CoT) chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts"],
        template=dedent(
            """
            You are an intelligent agent that is generating one thought at a time in
            a tree of thoughts setting.

            PROBLEM

            {{problem_description}}

            {% if thoughts %}
            THOUGHTS

            {% for thought in thoughts %}
            {{ thought }}
            {% endfor %}
            {% endif %}

            Let's think step by step.
            """
        ).strip(),
    )


def get_cot_prompt_ja() -> PromptTemplate:
    """Get the prompt for the Chain of Thought (CoT) chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts"],
        template=dedent(
            """
あなたは、思考ツリーの設定で思考を生成するインテリジェントなエージェントです。

            PROBLEM

            {{problem_description}}

            {% if thoughts %}
            THOUGHTS

            {% for thought in thoughts %}
            {{ thought }}
            {% endfor %}
            {% endif %}

ステップバイステップで考えてください。
            """
        ).strip(),
    )


def get_propose_prompt() -> PromptTemplate:
    """Get the prompt for the PROPOSE_PROMPT chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts", "n"],
        output_parser=JSONListOutputParser(),
        template=dedent(
            """
                You are an intelligent agent that is generating thoughts in a tree of
                thoughts setting.

                The output should be a markdown code snippet formatted as a JSON list of
                strings, including the leading and trailing "```json" and "```":

                ```json
                [
                "<thought-1>",
                "<thought-2>",
                "<thought-3>"
                ]
                ```

                PROBLEM

                {{ problem_description }}

                {% if thoughts %}
                VALID THOUGHTS

                {% for thought in thoughts %}
                {{ thought }}
                {% endfor %}

                Possible next {{ n }} valid thoughts based on the last valid thought:
                {% else %}

                Possible next {{ n }} valid thoughts based on the PROBLEM:
                {%- endif -%}
                """
        ).strip(),
    )


def get_propose_prompt_ja() -> PromptTemplate:
    """Get the prompt for the PROPOSE_PROMPT chain."""

    return PromptTemplate(
        template_format="jinja2",
        input_variables=["problem_description", "thoughts", "n"],
        output_parser=JSONListOutputParser(),
        template=dedent(
            """
あなたは、Three-of-Thought (ToT) で思考を生成するインテリジェントなエージェントです。

問題と、解決のための思考プロセス(VALID THOUGHTS)が確定しています。
VALID THOUGHTS に続く最新の思考候補を {{ n }} 個出力してください。
出力は、先頭に "```json"、末尾に "```" を含む、JSON形式の文字列リストとしてマークダウンコードスニペットにフォーマットしてください。

出力例を示します。
```json
[
    "<thought-1>",
    "<thought-2>",
    "<thought-3>"
]
問題は以下の通りです。
PROBLEM

{{ problem_description }}

{% if thoughts %}
VALID THOUGHTS(思考)

{% for thought in thoughts %}
{{ thought }}
{% endfor %}

上記の思考を参考にして、次の {{ n }} 個の最新の思考を出力してください。
{% else %}

上記の PROBLEM を参考にして、次の {{ n }} 個の最新の思考を出力してください。

{%- endif -%}

ガイドライン：

簡単な問題などで候補が完全に同じになってもかまいません。
思考を生成するために、問題と前回の思考を注意深く分析してください。
思考は、問題解決に向けた明確なステップや洞察を提供するものでなければなりません。
思考がループしている場合は、直ちに気づいて修正してください。あなたは頻繁に思考ループに陥る傾向があります。
各思考は簡潔にまとめ、問題に直接関連するようにしてください。
思考の質を向上させるために、必要に応じて問題をさらに分析し、追加の情報を検討してください。
より少ない思考で問題を解決できるように最善を尽くしてください。
生成された思考のリストを、指定されたJSON形式で出力してください。
                """
        ).strip(),
    )
