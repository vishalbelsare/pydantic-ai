from __future__ import annotations as _annotations

from dataclasses import dataclass

from ..tools import ToolDefinition
from . import AgentModel, KnownModelName, Model, infer_model


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_stack: list[Model]
    current_model_idx: int

    def __init__(
        self,
        default_model: Model | KnownModelName,
        *fallback_models: Model | KnownModelName,
        # TODO: do we need any of the other classic init args (base url, etc?)
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
        """
        # TODO: should we do this lazily?
        default_model_ = default_model if isinstance(default_model, Model) else infer_model(default_model)
        fallback_models_ = [model if isinstance(model, Model) else infer_model(model) for model in fallback_models]

        self.model_stack = [default_model_, *fallback_models_]
        self.current_model_idx = 0

    @property
    def current_model(self) -> Model:
        return self.model_stack[self.current_model_idx]

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        # TODO: still need to implement actual fallback model logic
        return await self.current_model.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )

    def name(self) -> str:
        return self.current_model.name()
