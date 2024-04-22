from interpreter.core.core import OpenInterpreter
from ui.server_impl import server

interpreter = OpenInterpreter(
    auto_run=True,
)
print("interpreter.sync_computer=", interpreter.sync_computer)
interpreter.llm.model = "claude-3-haiku-20240307"
interpreter.llm.max_tokens = 2000
interpreter.verbose = True
interpreter.sync_computer = True
interpreter.system_message += """
Your workdir is "/app/work". You should save any input and output files in this directory.
If you lost previous work, you should check this directory and result from files.
"""
print("system_message=", interpreter.system_message)

computer = interpreter.computer
computer.debug = True
computer.verbose = True
server(interpreter)
