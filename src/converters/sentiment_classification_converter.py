from converters.registry import register_converter
from converters.source_target_converter import SourceTargetConverter

@register_converter('sentiment')
class SentimentConverter(SourceTargetConverter):
    """
    ```
    >>> from converters.registry import get_converter
    >>> converter = get_converter('sentiment')
    >>> print(converter.example2code(demos, target))
    INSTRUCTION: Analyze the sentiment of the following text excerpts, categorizing them as either 'positive',
    or 'negative'.
    source: I love this place, the scenery is breathtaking and people are so friendly.
    target: positive
    source: The customer service was terrible, I waited for hours.
    target: negative
    source: This is the best day of my life!
    target:
    ```
    """
    INSTRUCTION = "Analyze the sentiment of the following text excerpts, categorizing them as either 'positive', or " \
                  "'negative'.\n"
    LABEL_MAP = {0: "negative", 1: "positive"}

    def string2label(self, answer: int):
        # convert string in answer into label
        return self.LABEL_MAP[answer]