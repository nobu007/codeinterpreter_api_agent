from typing import Any, Optional, Type

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from codeinterpreterapi.brain.params import CodeInterpreterParams


class CustomFunctionInput(BaseModel):
    question: str = Field(description="the original question to response user.")
    message: str = Field(description="response message from this tool.")


class AgentWrapperTool(BaseTool):
    """Tool that wraps an agent and exposes it as a LangChain tool."""

    agent_impl: Optional[Any] = None
    args_schema: Type[BaseModel] = CustomFunctionInput

    @classmethod
    def create_agent_wrapper_tools(cls, ci_params: CodeInterpreterParams) -> None:
        # 各エージェントに対してノードを作成
        tools = []
        for agent_def in ci_params.agent_def_list:
            agent = agent_def.agent_executor
            agent_name = agent_def.agent_name
            agent_role = agent_def.agent_role
            tool = cls(agent_name=agent_name, agent_role=agent_role, agent_impl=agent)
            tools.append(tool)

        return tools

    def __init__(self, agent_name: str, agent_role: str, agent_impl: Any):
        super().__init__(
            name=agent_name,
            description=f"A tool that wraps the {agent_name} agent with role: {agent_role}. "
            f"Input should be a query or task for the {agent_name} agent.",
        )
        self.agent_impl = agent_impl

    def _run(
        self,
        question: str,
        message: str,
    ) -> str:
        """Use the tool."""
        messages = []
        # messages.append(HumanMessage(question))
        messages.append(AIMessage(message))
        return self.agent_impl.invoke({"input": question, "messages": messages})

    async def _arun(
        self,
        question: str,
        message: str,
    ) -> str:
        """Use the tool."""
        messages = []
        messages.append(HumanMessage(question))
        messages.append(AIMessage(message))
        return await self.agent_impl.ainvoke(messages)
