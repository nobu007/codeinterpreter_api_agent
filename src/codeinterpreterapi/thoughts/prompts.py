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
あなたは、思考ツリーの設定で思考を生成するインテリジェントなエージェントです。

出力は、先頭に "```json"、末尾に "```" を含む、JSON形式の文字列リストとしてマークダウンコードスニペットにフォーマットしてください。

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

次の思考を生成するために、問題と有効な思考を注意深く分析してください。
思考は、問題解決に向けた明確なステップや洞察を提供するものでなければなりません。
各思考は簡潔にまとめ、問題に直接関連するようにしてください。
思考の質を向上させるために、必要に応じて問題をさらに分析し、追加の情報を検討してください。
生成された思考のリストを、指定されたJSON形式で出力してください。
                """
        ).strip(),
    )
