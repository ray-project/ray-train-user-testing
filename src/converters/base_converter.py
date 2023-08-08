from typing import List
from dataset import Example

class BaseConverter:
    INSTRUCTION = ""
    # define a static function that is used to convert answer into label
    LABEL_MAP = {}
    def string2label(self, answer: str):
        raise NotImplementedError()

    def example2code(self, demos: List[Example], target: Example) -> str:
        raise NotImplementedError()

    def code2answer(self, code: str) -> str:
        raise NotImplementedError()