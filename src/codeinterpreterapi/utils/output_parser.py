from typing import Union

from langchain.agents.conversational.output_parser import ConvoOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException


class PlannerMultiOutputParser(ConvoOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into AgentAction, AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            # TODO update
            return super().parse(text)
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
            raise OutputParserException(f"Could not parse LLM output: {text}") from e


class PlannerSingleOutputParser(ConvoOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        print("PlannerSingleOutputParser text=", text)
        final_keyword = "<END_OF_PLAN>"
        if final_keyword in text:
            return AgentFinish({"output": text.split(final_keyword)[0]}, text)
        else:
            try:
                return super().parse(text)
            except Exception as e:
                # If any other exception is raised during parsing, also raise an
                # OutputParserException
                raise OutputParserException(f"Could not parse LLM output: {text}") from e
