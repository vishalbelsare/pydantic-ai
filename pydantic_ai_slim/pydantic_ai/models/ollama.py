from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal, Union

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from e


from .openai import OpenAIModel

CommonOllamaModelNames = Literal[
    'codellama',
    'deepseek-r1',
    'gemma',
    'gemma2',
    'llama3',
    'llama3.1',
    'llama3.2',
    'llama3.2-vision',
    'llama3.3',
    'mistral',
    'mistral-nemo',
    'mixtral',
    'phi3',
    'phi4',
    'qwq',
    'qwen',
    'qwen2',
    'qwen2.5',
    'starcoder2',
]
"""This contains just the most common ollama models.

For a full list see [ollama.com/library](https://ollama.com/library).
"""
OllamaModelName = Union[CommonOllamaModelNames, str]
"""Possible ollama models.

Since Ollama supports hundreds of models, we explicitly list the most models but
allow any name in the type hints.
"""


@dataclass(init=False)
class OllamaModel(OpenAIModel):
    """A model that implements Ollama using the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the Ollama server.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def name(self) -> str:
        return f'ollama:{self.model_name}'
