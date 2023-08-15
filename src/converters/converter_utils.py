import importlib
import inspect
from typing import Type, Dict, Tuple, Callable
import ast
import re
import os


def to_camel_case(s: str) -> str:
    """
    Example:
        'ocr_tokens' -> 'OcrTokens'
        'ocr tokens' -> 'OcrTokens'
    """
    s = '_'.join(s.split())
    return ''.join([w.capitalize() for w in s.split('_')])


def to_snake_case(s: str) -> str:
    """
    Example:
        'OcrTokens' -> 'ocr_tokens'
        'ocr tokens' -> 'ocr_tokens'
    """
    if s.islower():
        return '_'.join(s.split())
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_whitespaced(s: str) -> str:
    """
    Example:
        'OcrTokens' -> 'ocr tokens'
        'ocr_tokens' -> 'ocr tokens'
    """
    if not s.islower():
        s = to_snake_case(s)
    return ' '.join(s.split('_'))


def compile_code_get_object(py_code_str: str):
    """
    adapted from https://github.com/reasoning-machines/CoCoGen/blob/main/src/converters/utils.py
    """
    # compile the code
    try:
        py_code = compile(py_code_str, "<string>", "exec")
    except SyntaxError:
        # try without the last k lines in py_code_str: usually the last line is incomplete
        for k in range(1, 2):
            try:
                lines = py_code_str.split("\n")
                lines = "\n".join(lines[:-k])
                py_code = compile(lines, "<string>", "exec")
            except SyntaxError as e:
                print(f"Error compiling python code:\n{py_code_str}")
                raise e

    # instantiate the class
    py_code_dict = {}
    exec(py_code, py_code_dict)
    # the newly instantiated class will be last in the scope
    py_code_class = py_code_dict[list(py_code_dict.keys())[-1]]
    return py_code_class


def setup_imports():
    import os
    print("current_directory", os.getcwd())
    converter_modules = [f'converters.{f.replace(".py", "")}'
                         for f in os.listdir('converters') if f.endswith('.py')
                         and f not in ('registry.py', 'convertor_utils.py')]
    for module in converter_modules:
        importlib.import_module(module)