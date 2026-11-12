"""The cores and the memory of a run in its progress."""

from __future__ import annotations

import io
import json
import logging
import sys
import time

import pytest
from rich.console import Console

from sbmlsim import events, progress
from sbmlsim.progress import resources

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc")


def _busy(seconds: float = 0.2) -> None:
    """Burn the CPU of the process for a while."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sum(i * i for i in range(10000))


def test_a_sample_names_the_cores_of_the_run():
    sample = resources.Sampler().sample(workers=4)
    assert sample.cores >= 1
    assert sample.workers == 4
    assert "cores 4/" in resources.format_resources(sample)


def test_the_cpu_percent_is_measured_between_two_samples(clock):
    sampler = resources.Sampler(now=clock)
    assert sampler.sample().cpu_percent is None
    clock.tick(2.0)
    _busy(0.2)
    sample = sampler.sample()
    assert sample.cpu_percent is not None
    assert sample.cpu_percent > 0.0
    assert "cpu" in resources.format_resources(sample)


def test_a_sample_inside_the_interval_is_the_one_before(clock):
    sampler = resources.Sampler(now=clock)
    first = sampler.sample(interval=10.0)
    clock.tick(1.0)
    assert sampler.sample(interval=10.0) is first
    clock.tick(10.0)
    assert sampler.sample(interval=10.0) is not first


def test_a_sample_inside_the_interval_takes_the_workers_of_the_call(clock):
    sampler = resources.Sampler(now=clock)
    sampler.sample(interval=10.0, workers=1)
    assert sampler.sample(interval=10.0, workers=8).workers == 8


@linux
def test_the_memory_of_the_run_is_that_of_its_process_tree():
    sample = resources.Sampler().sample()
    assert sample.rss is not None
    assert sample.rss > 0
    assert sample.memory_total is not None
    assert sample.memory_used is not None
    assert sample.memory_used <= sample.memory_total
    text = resources.format_resources(sample)
    assert "mem" in text
    assert "machine" in text


def test_a_sample_without_proc_leaves_the_memory_out(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "PROC", tmp_path / "nothing")
    sample = resources.Sampler().sample()
    assert sample.rss is None
    assert sample.memory_total is None
    assert "mem" not in resources.format_resources(sample)
    assert "cores" in resources.format_resources(sample)


def test_the_dictionary_of_a_sample_is_json(clock):
    sample = resources.Sampler(now=clock).sample(workers=2)
    assert json.loads(json.dumps(sample.as_dict()))["workers"] == 2


def test_the_log_line_names_the_cores_and_the_memory(caplog, clock, monkeypatch):
    monkeypatch.setattr(progress.core, "_now", clock)
    caplog.set_level(logging.INFO, logger="sbmlsim.progress")
    with (
        progress.reporter(progress.LogReporter(interval=0.0, now=clock)),
        progress.activity("fit", "NMP", total=2, workers=3) as a,
    ):
        a.advance()
    assert "cores 3/" in caplog.records[0].message


def test_the_event_carries_the_resources(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(progress.core, "_now", clock)
    path = tmp_path / "progress.jsonl"
    with (
        events.recording(path),
        progress.reporter(progress.EventReporter(throttle=0.0, now=clock)),
        progress.activity("scan", total=2, workers=2) as a,
    ):
        a.advance()
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["resources"]["cores"] >= 1
    assert records[-1]["resources"]["workers"] == 2


def test_the_bar_of_the_outermost_activity_names_the_resources():
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=160)
    with (
        progress.reporter(progress.BarReporter(console=console)),
        progress.activity("fit", "NMP", total=2, workers=2) as run,
    ):
        run.advance()
        with run.child("jacobian", total=4) as jacobian:
            jacobian.advance()
    lines = [line for line in out.getvalue().splitlines() if "jacobian" in line]
    assert lines
    assert all("cores" not in line for line in lines)
    assert "cores" in out.getvalue()
