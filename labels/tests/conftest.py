"""Shared fixtures: a fake inferer that answers every chat message with a callable."""
import pytest


class FakeInferer:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def prompt_fits(self, messages, max_new_tokens, *, safety_tokens=0):
        return True, 0

    def make_sampling(self, **kw):
        return kw

    def chat(self, batch, sampling):
        self.calls.append((batch, sampling))
        return [self.answer(m) for m in batch]


@pytest.fixture
def fake_inferer():
    return FakeInferer
