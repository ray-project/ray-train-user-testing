from typing import Dict, Type
from converters.base_converter import BaseConverter

_converter_mapping: Dict[str, Type[BaseConverter]] = {}


def register_converter(name: str):
    def wrapper(cls):
        if not issubclass(cls, BaseConverter):
            raise ValueError('All converters must inherit from BaseConverter class')
        _converter_mapping[name] = cls
        return cls

    return wrapper


def get_converter(name: str) -> BaseConverter:
    if name not in _converter_mapping:
        raise ValueError(f'Converter {name} not registered')
    return _converter_mapping[name]()