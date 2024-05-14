from codeinterpreterapi import CodeInterpreterSession


def test():
    model = "claude-3-haiku-20240307"
    # model = "gemini-1.0-pro"
    test_message = "pythonで円周率を表示するプログラムを実行してください。"
    verbose = False
    is_streaming = False
    print("test_message=", test_message)
    session = CodeInterpreterSession(model=model, verbose=verbose)
    status = session.start_local()
    print("status=", status)
    if is_streaming:
        # response_inner: CodeInterpreterResponse
        response_inner = session.generate_response_stream(test_message)
        for response_inner_chunk in response_inner:
            print("response_inner_chunk.content=", response_inner.content)
            print("response_inner_chunk code_log=", response_inner.code_log)
    else:
        # response_inner: CodeInterpreterResponse
        response_inner = session.generate_response(test_message)
        print("response_inner.content=", response_inner.content)
        print("response_inner code_log=", response_inner.code_log)


if __name__ == "__main__":
    test()
