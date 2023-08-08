from converters.registry import register_converter
from converters.source_target_converter import SourceTargetConverter

@register_converter('semantic_parsing')
class SemanticParsingConverter(SourceTargetConverter):
    """
    ```
    >>> from converters.registry import get_converter
    >>> converter = get_converter('semantic_parsing')
    >>> print(converter.example2code(demos, target))
    INSTRUCTION: Mapping the natural language question to the corresponding SQL query.\n

    source: What are the major cities in states through which the mississippi runs?\n

    target: major(city(loc_2( state(traverse_1(riverid('mississippi'))))))\n
    ```
    """
    INSTRUCTION = "Mapping the natural language question to the corresponding SQL query.\n"