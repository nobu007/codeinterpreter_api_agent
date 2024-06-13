from gui_agent_loop_core.connector_impl.core_to_agent.connector_impl_codeinterpreter_api import (
    ConnectorImplCodeinterpreterApi,
)
from gui_agent_loop_core.gui_agent_loop_core import GuiAgentLoopCore

from codeinterpreterapi import CodeInterpreterSession


class CodeInterpreter(ConnectorImplCodeinterpreterApi):
    def __init__(self):
        # model = "claude-3-haiku-20240307"
        # model = "gemini-1.5-pro-latest"
        model = "gemini-1.5-flash-latest"
        # model = "gemini-1.0-pro"
        self.session = CodeInterpreterSession(model=model, verbose=True)
        self.status = self.session.start_local()


interpreter = CodeInterpreter()
core = GuiAgentLoopCore()
core.launch_server(interpreter)
