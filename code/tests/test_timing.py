"""Wall-clock helpers: call counts, call order and the statistic each returns."""

from __future__ import annotations

import itertools

import pytest
import timing


@pytest.fixture
def fake_clock(monkeypatch) -> None:
    """``perf_counter`` readings 1, 4, 6, 11, 15, 22, ... (steps 1, 3, 2, 5, 4, 7)."""
    steps = itertools.cycle([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    state = {"t": 0.0}

    def tick():
        state["t"] += next(steps)
        return state["t"]

    monkeypatch.setattr(timing.time, "perf_counter", tick)


def test_timed_returns_output_and_elapsed(fake_clock) -> None:
    out, seconds = timing.timed(lambda a, b=0: a + b, 2, b=3)
    assert out == 5
    assert seconds == 3.0


def test_mean_call_time_is_one_block_over_n_calls(fake_clock) -> None:
    calls = []
    assert timing.mean_call_time(lambda: calls.append(1), 4) == 3.0 / 4
    assert len(calls) == 4


def test_call_times_times_each_call_separately(fake_clock) -> None:
    calls = []
    assert timing.call_times(lambda: calls.append(1), 3) == [3.0, 5.0, 7.0]
    assert len(calls) == 3
