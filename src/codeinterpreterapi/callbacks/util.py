import sys
from copy import deepcopy
from typing import Any, Dict, List, Union


def get_current_function_name(depth: int = 1) -> str:
    return sys._getframe(depth).f_code.co_name


def show_callback_info(name: str, tag: str, data: Any) -> None:
    current_function_name = get_current_function_name(2)
    print("show_callback_info current_function_name=", current_function_name, name)
    print(f"{tag}=", trim_data(data))


def trim_data(data: Union[Any, List[Any], Dict[str, Any]]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :data: 対象データ
    """
    data_copy = deepcopy(data)
    return trim_data_iter("", data_copy)


def trim_data_iter(indent: str, data: Union[Any, List[Any], Dict[str, Any]]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param data: 対象データ
    """
    indent_next = indent + "  "
    if isinstance(data, dict):
        return trim_data_dict(indent_next, data)
    elif isinstance(data, list):
        return trim_data_array(indent_next, data)
    else:
        return trim_data_other(indent, data)


def trim_data_dict(indent: str, data: Dict[str, Any]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    new_data_list = []
    for k, v in data.items():
        new_data_list.append(f"{indent}dict[{k}]: " + trim_data_iter(indent, v))
    return "\n".join(new_data_list)


def trim_data_array(indent: str, data: List[Any]) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    new_data_list = []
    for i, item in enumerate(data):
        print(f"{indent}array[{str(i)}]: ")
        new_data_list.append(trim_data_iter(indent, item))
    return "\n".join(new_data_list)


def trim_data_other(indent: str, data: Any) -> str:
    """
    dataの構造をデバッグ表示用に短縮する関数

    :param indent: インデント文字列
    :param data: 対象データ
    """
    stype = str(type(data))
    s = str(data)
    return f"{indent}type={stype}, data={s[:80]}"
