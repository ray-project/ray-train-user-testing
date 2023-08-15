from typing import List
from dataset import Example
from converters.base_converter import BaseConverter
# from converters.registry import register_converter
from typing import Optional

# @register_converter('source_target')
class SourceTargetConverter(BaseConverter):
    """
    ```
    >>> from converters.registry import get_converter
    >>> converter = get_converter('source_target')
    >>> print(converter.example2code(demos, target))
    source: what is the brand of this camera?
    target: dakota
    source: what is the brand of this camera?
    target:
    ```
    """

    def example2code(self, demos: List[Example], target: Example) -> str:
        rst = self.INSTRUCTION + "\n" #+ sep_token
        for example in demos:
            rst += f"source: {example.source_input}\n"
            if example.target_label in self.LABEL_MAP:
                rst += f"target: {self.LABEL_MAP[example.target_label]}\n"
            else:
                rst += f"target: {example.target_label}\n"
            # rst += sep_token
        rst += f"source: {target.source_input}\n"
        rst += f"target:" #+ sep_token
        return rst

    def code2answer(self, code: str) -> str:
        lines = code.strip().split('\n')
        targets = [line for line in lines if line.startswith('target')]
        return targets[-1].replace('target:', '').strip()

    def string2label(self, answer: str):
        # convert string in answer into label
        return answer