import asyncio

from invoke import Context, task


@task
def run_code(c, code):
    """
    LLMから生成されたPythonコードを非同期で実行し、結果をパースする
    """
    loop = asyncio.get_event_loop()
    exitcode, stdout, stderr = loop.run_until_complete(execute_code(code))
    ret = ""
    ret += f"Exit Code: {exitcode}\n"
    ret += f"Output: \n{stdout.decode()}\n"
    if stderr:
        ret += f"Error: \n{stderr.decode()}\n"

    # 一時ファイルを削除
    c.run("rm temp.py")

    return ret


@task
def run_code_file(c, code_file):
    """
    LLMから生成されたPythonコードを保存したファイルを非同期で実行し、結果をパースする
    """
    loop = asyncio.get_event_loop()
    exitcode, stdout, stderr = loop.run_until_complete(execute_code_file(code_file))
    ret = ""
    ret += f"Exit Code: {exitcode}\n"
    ret += f"Output: \n{stdout.decode()}\n"
    if stderr:
        ret += f"Error: \n{stderr.decode()}\n"
    print(ret)

    return ret


async def execute_code(code):
    """
    Pythonコードを非同期で実行する
    """
    # コードを一時ファイルに保存
    with open("temp.py", "w") as f:
        f.write(code)

    # Pythonコードを非同期で実行し、出力をキャプチャ
    proc = await asyncio.create_subprocess_exec(
        "python",
        "temp.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    exitcode = proc.returncode

    return exitcode, stdout, stderr


async def execute_code_file(code_file):
    """
    Pythonファイルを非同期で実行する
    """

    # Pythonコードを非同期で実行し、出力をキャプチャ
    proc = await asyncio.create_subprocess_exec(
        "python",
        code_file,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    exitcode = proc.returncode

    return exitcode, stdout, stderr


@task
def main(c, code=None):
    """
    LLMから生成されたPythonコードを実行・修正を繰り返す
    """
    loop = asyncio.get_event_loop()

    try:
        print("Current code:")
        print(code)

        exitcode, stdout, stderr = loop.run_until_complete(execute_code(code))
        ret = ""
        ret += f"Exit Code: {exitcode}\n"
        ret += f"Output: \n{stdout.decode()}\n"
        ret += f"Error: \n{stderr.decode()}\n"
        print(f"Exit Code: {exitcode}")
        print(f"Output: \n{stdout.decode()}")
        print(f"Error: \n{stderr.decode()}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        c.run("rm temp.py")


# execute_code コルーチンは前と同様

if __name__ == "__main__":
    # 初期コードを引数で受け取る (オプション)
    initial_code = "print('ddd')"
    run_code(Context(), code=initial_code)
    main(Context(), code=initial_code)
