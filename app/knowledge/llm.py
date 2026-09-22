from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

__all__ = ["StructuredLLM", "GroqLLM", "FakeLLM", "DEFAULT_MODEL"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "openai/gpt-oss-120b"


@runtime_checkable
class StructuredLLM(Protocol):
    """The single seam every LLM call goes through, so tests can run offline."""

    def structured(self, prompt: str, schema: type[T]) -> T: ...


class GroqLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        from langchain_groq import ChatGroq

        from app.config import settings

        self.model = model
        self._llm = ChatGroq(model=model, api_key=api_key or settings.groq_api_key)

    def structured(self, prompt: str, schema: type[T]) -> T:
        return self._llm.with_structured_output(schema).invoke(prompt)


class FakeLLM:
    """Deterministic stand-in. Queue responses, or answer from a handler.

    A queued `Exception` is raised rather than returned, so failure paths are
    testable without patching.
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        handler: Callable[[str, type[BaseModel]], Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self.prompts: list[str] = []

    def structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if self._handler is not None:
            return self._handler(prompt, schema)
        assert self._responses, f"no queued response for {schema.__name__}"
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
