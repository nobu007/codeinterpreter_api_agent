from typing import Union

from langchain.agents.conversational.output_parser import ConvoOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException


class CustomOutputParser(ConvoOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            # If the response contains an 'action' and 'action_input'

            print("CustomOutputParser text=", text)
            if "Action" in text or "Action Input" in text or "Final Answer" in text:
                # If the action indicates a final answer, return an AgentFinish
                if "<END_OF_PLAN>" in text:
                    return AgentFinish({"output": text.split('Final Answer:')[1]}, text)
                else:
                    return super().parse(text)
            else:
                # If the necessary keys aren't present in the response, raise an
                # exception
                raise OutputParserException(f"Missing 'action' or 'action_input' in LLM output: {text}")
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
            raise OutputParserException(f"Could not parse LLM output: {text}") from e
