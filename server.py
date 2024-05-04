from codeinterpreterapi import CodeInterpreterSession

model = "claude-3-haiku-20240307"
session = CodeInterpreterSession(model=model)

try:
    status = session.start_local()
    result = session.generate_response(
        "Plot the nvidea stock vs microsoft stock over the last 6 months."
    )
    result.show()
except Exception as e:
    print(e)
