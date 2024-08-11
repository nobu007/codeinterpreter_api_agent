import ast
import os
import tempfile

from codeinterpreterapi.config import settings


class FileUtil:
    @staticmethod
    def get_python_file_path(filename: str) -> str:
        python_file_dir = settings.TEMP_DIR
        python_file_dir_work = settings.WORK_DIR
        if os.path.isdir(python_file_dir_work):
            python_file_dir = python_file_dir_work
        python_file_path = os.path.join(python_file_dir, filename)
        return python_file_path

    @staticmethod
    def write_python_file(code: str):
        if FileUtil.is_raw_string(code):
            print("FileUtil write_python_file raw string by ast.parse()")
            parsed_code = ast.parse(code)
            code_content = ast.unparse(parsed_code)
        else:
            print("FileUtil write_python_file regular string by ast.literal_eval()")
            code_content = ast.literal_eval(f'"""{code}"""')

        if settings.PYTHON_OUT_FILE:
            python_file_path = FileUtil.get_python_file_path(filename=settings.PYTHON_OUT_FILE)
            with open(python_file_path, "w", encoding="utf-8") as python_file:
                python_file.write(code_content)
                return python_file_path
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", dir=settings.WORK_DIR) as temp_file:
                temp_file.write(code_content)
                temp_file_path = temp_file.name
                return temp_file_path

    @staticmethod
    def read_python_file(filename: str = None) -> str:
        print("FileUtil read_python_file filename=", filename)
        if filename is None:
            filename = settings.PYTHON_OUT_FILE
        python_file_path = FileUtil.get_python_file_path(filename=filename)
        with open(python_file_path, encoding="utf8") as f:
            latest_code = f.readlines()
            latest_code = "\n".join(latest_code)
            return latest_code
        return "no python_file_path=" + python_file_path

    @staticmethod
    def is_raw_string(code: str) -> str:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
