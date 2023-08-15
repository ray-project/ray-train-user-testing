# from converters.registry import register_converter
from converters.source_target_converter import SourceTargetConverter

# @register_converter('math')
class MathConverter(SourceTargetConverter):
    """
    ```
    >>> from converters.registry import get_converter
    >>> converter = get_converter('math')
    >>> print(converter.example2code(demos, target))
    INSTRUCTION: Solve the following math problems.

    source: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
    How many clips did Natalia sell altogether in April and May?

    target: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>
    72 clips altogether in April and May. #### 72
    ```
    """
    INSTRUCTION = "Solve the following math problems.\n"

    def string2label(self, answer: str):
        # convert string in answer into label
        return answer.split("####")[-1].strip()

    def code2answer(self, code: str) -> str:
        lines = code.strip().split('\n')
        targets = [line for line in lines if line.startswith('####')]
        last_target = targets[-1]
        # get the number in the last target seperated by ####
        return last_target.replace('####', '').strip()