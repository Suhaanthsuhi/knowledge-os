from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.knowledge.llm import FakeLLM


class Shape(BaseModel):
    value: str


def test_fake_llm_returns_queued_responses_in_order():
    llm = FakeLLM(responses=[Shape(value="one"), Shape(value="two")])
    assert llm.structured("a", Shape).value == "one"
    assert llm.structured("b", Shape).value == "two"


def test_fake_llm_records_every_prompt():
    llm = FakeLLM(responses=[Shape(value="one")])
    llm.structured("the prompt", Shape)
    assert llm.prompts == ["the prompt"]


def test_fake_llm_raises_when_exhausted():
    llm = FakeLLM(responses=[])
    with pytest.raises(AssertionError, match="no queued response"):
        llm.structured("a", Shape)


def test_fake_llm_can_answer_from_a_handler():
    llm = FakeLLM(handler=lambda prompt, schema: schema(value=prompt.upper()))
    assert llm.structured("hi", Shape).value == "HI"


def test_fake_llm_can_raise_a_queued_error():
    llm = FakeLLM(responses=[RuntimeError("upstream exploded")])
    with pytest.raises(RuntimeError, match="upstream exploded"):
        llm.structured("a", Shape)
