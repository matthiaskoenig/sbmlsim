import json
from pathlib import Path
from typing import Any

import pytest

from sbmlsim import run
from sbmlsim.run import ConfigurationError, PlanMismatch
from sbmlsim.run.units import MissingUnits
from tests.run.kinds_dummy import DummySpec

DUMMY = "tests.run.kinds_dummy:DUMMY"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_simrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of the tests is no task of simrun, whatever the environment."""
    import os

    for name in list(os.environ):
        if name.startswith("SIMRUN_"):
            monkeypatch.delenv(name)


def test_execute_writes_spec_run_summary_progress(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(DUMMY, {"n": 10, "seed": 2}, out)

    assert summary["total"] == sum((2 + i) ** 2 for i in range(10))
    assert _json(out / "summary.json") == summary
    assert _json(out / "total.json") == summary["total"]
    planned = DummySpec.model_validate(_json(out / "spec.json"))
    assert planned.units == 10
    assert planned.seed == 2
    record = _json(out / "run.json")
    assert record["kind"] == "dummy"
    assert record["status"] == "done"
    assert record["spec_hash"] == run.spec_hash(DummySpec(n=10, seed=2), ("fit",))
    assert record["versions"]["sbmlsim"]
    assert record["python"]
    assert record["host"]
    assert record["workers"] == 1
    assert record["started"] <= record["finished"]
    assert record["sources"] is None
    assert record["git"] is None
    lines = (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["index"] for line in lines] == list(range(10))


def test_run_json_records_sources_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [{"name": "model", "sha": "abc", "dirty": False}]
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(sources), encoding="utf-8")
    monkeypatch.setenv("SIMRUN_PROVENANCE", str(provenance))
    monkeypatch.setenv("SIMRUN_ATTEMPT", "1")
    monkeypatch.setenv("SIMRUN_TOKEN", "secret")
    monkeypatch.setenv("SIMRUN_STORE_KEY", "secret")
    monkeypatch.setenv("SIMRUN_URL", "https://user:secret@simrun")

    out = tmp_path / "out"
    run.execute(
        DUMMY,
        {"n": 3, "project": "examples.hctz_fitting.project:PROJECT"},
        out,
        workers=2,
    )

    record = _json(out / "run.json")
    assert _json(out / "summary.json")["workers"] == 2
    assert record["sources"] == sources
    assert record["environment"] == {
        "SIMRUN_ATTEMPT": "1",
        "SIMRUN_PROVENANCE": str(provenance),
    }
    assert "secret" not in (out / "run.json").read_text(encoding="utf-8")
    assert record["git"] is None
    assert record["workers"] == 2


def test_run_json_records_git_without_sources(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute(
        DUMMY, {"n": 3, "project": "examples.hctz_fitting.project:PROJECT"}, out
    )

    record = _json(out / "run.json")
    assert record["sources"] is None
    assert len(record["git"]["sha"]) == 40
    assert record["environment"] == {}


def test_provenance_argument_is_read(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps([{"name": "a"}]), encoding="utf-8")
    out = tmp_path / "out"
    run.execute(DUMMY, {"n": 2}, out, provenance=provenance)
    assert _json(out / "run.json")["sources"] == [{"name": "a"}]


def test_invalid_provenance_is_a_configuration_error(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="provenance"):
        run.execute(DUMMY, {"n": 2}, tmp_path / "out", provenance=provenance)


def test_failed_run_is_recorded(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="the dummy kind fails"):
        run.execute(DUMMY, {"n": 2, "fail": True}, out)
    record = _json(out / "run.json")
    assert record["status"] == "failed"
    assert "the dummy kind fails" in record["error"]
    assert record["finished"]
    assert not (out / "summary.json").exists()


def test_out_not_empty_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "other.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not empty"):
        run.execute(DUMMY, {"n": 2}, out)


def test_out_which_is_a_file_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no directory"):
        run.execute(DUMMY, {"n": 2}, out)


def test_resume_uses_the_stored_spec(tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = run.execute(DUMMY, {"n": 5}, out)
    seed = _json(out / "spec.json")["seed"]
    (out / "parts" / "3.json").unlink()
    (out / "summary.json").unlink()

    second = run.execute(DUMMY, {"n": 5}, out, resume=True)

    assert _json(out / "spec.json")["seed"] == seed
    assert second == first
    assert _json(out / "run.json")["status"] == "done"


def test_resume_into_an_empty_out_runs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run.execute(DUMMY, {"n": 2, "seed": 1}, out, resume=True)
    assert summary["total"] == 1 + 4


def test_resume_without_a_record_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "other.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"run\.json"):
        run.execute(DUMMY, {"n": 2}, out, resume=True)


def test_resume_with_another_spec_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute(DUMMY, {"n": 4}, out)
    with pytest.raises(ConfigurationError, match="another spec") as info:
        run.execute(DUMMY, {"n": 5}, out, resume=True)
    # the planned spec of the out is no spec the run was started with
    assert "with the spec file the run was started with" in str(info.value)
    assert "not its planned spec 'spec.json'" in str(info.value)
    planned = _json(out / "spec.json")
    with pytest.raises(ConfigurationError, match="another spec"):
        run.execute(DUMMY, planned, out, resume=True)


def test_split_equals_execute(tmp_path: Path) -> None:
    planned = run.plan(DUMMY, {"n": 130})
    assert isinstance(planned, DummySpec)
    assert planned.seed is not None
    n_units = run.count_units(DUMMY, planned)
    assert n_units == 44

    split = tmp_path / "split"
    for index in range(n_units):
        run.compute_unit(DUMMY, planned, index, split)
    assert not (split / "run.json").exists()
    assert (split / "progress" / "unit-43.jsonl").exists()
    summary = run.merge(DUMMY, planned, split)

    whole = tmp_path / "whole"
    assert run.execute(DUMMY, planned, whole) == summary
    assert (split / "total.json").read_bytes() == (whole / "total.json").read_bytes()
    record = _json(split / "run.json")
    assert record["status"] == "done"
    assert record["spec_hash"] == run.spec_hash(planned, ("fit",))
    assert _json(split / "summary.json") == summary
    assert DummySpec.model_validate(_json(split / "spec.json")) == planned


def test_unit_twice_is_idempotent(tmp_path: Path) -> None:
    planned = run.plan(DUMMY, {"n": 10})
    run.compute_unit(DUMMY, planned, 4, tmp_path)
    content = (tmp_path / "parts" / "4.json").read_bytes()
    run.compute_unit(DUMMY, planned, 4, tmp_path)
    assert (tmp_path / "parts" / "4.json").read_bytes() == content
    assert sorted(p.name for p in (tmp_path / "parts").iterdir()) == ["4.json"]


def test_unit_outside_the_range_is_refused(tmp_path: Path) -> None:
    planned = run.plan(DUMMY, {"n": 10})
    for index in [-1, 10]:
        with pytest.raises(ConfigurationError, match="range"):
            run.compute_unit(DUMMY, planned, index, tmp_path)


def test_merge_names_missing_units(tmp_path: Path) -> None:
    planned = run.plan(DUMMY, {"n": 10})
    for index in range(10):
        if index not in (2, 7):
            run.compute_unit(DUMMY, planned, index, tmp_path)
    with pytest.raises(MissingUnits, match="2, 7") as info:
        run.merge(DUMMY, planned, tmp_path)
    assert info.value.missing == (2, 7)
    assert _json(tmp_path / "run.json")["status"] == "failed"


def test_count_units_needs_a_plan() -> None:
    with pytest.raises(ConfigurationError, match="plan"):
        run.count_units(DUMMY, {"n": 10})
    assert run.count_units(DUMMY, run.plan(DUMMY, {"n": 10})) == 10


def test_unit_and_merge_need_a_plan(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="plan"):
        run.compute_unit(DUMMY, {"n": 10}, 0, tmp_path)
    with pytest.raises(ConfigurationError, match="plan"):
        run.merge(DUMMY, {"n": 10}, tmp_path)


def test_relative_paths_are_relative_to_the_spec_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "specs"
    directory.mkdir()
    spec = directory / "spec.yaml"
    spec.write_text("n: 2\ndata: data/values.csv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    planned = run.plan(DUMMY, spec)

    assert isinstance(planned, DummySpec)
    assert planned.data == directory.resolve() / "data" / "values.csv"


def test_spec_files_of_json_and_yaml(tmp_path: Path) -> None:
    as_json = tmp_path / "spec.json"
    as_json.write_text(json.dumps({"n": 3, "seed": 1}), encoding="utf-8")
    as_yaml = tmp_path / "spec.yml"
    as_yaml.write_text("n: 3\nseed: 1\n", encoding="utf-8")
    assert run.plan(DUMMY, as_json) == run.plan(DUMMY, as_yaml)


@pytest.mark.parametrize(
    ("content", "match"), [("- 1\n- 2\n", "mapping"), ("n: [1\n", "YAML")]
)
def test_spec_file_which_is_no_mapping_is_refused(
    tmp_path: Path, content: str, match: str
) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=match):
        run.plan(DUMMY, spec)


def test_missing_spec_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="spec"):
        run.plan(DUMMY, tmp_path / "missing.yaml")


def test_inputs_set_input_fields(tmp_path: Path) -> None:
    fit = tmp_path / "fit"
    fit.mkdir()
    out = tmp_path / "out"
    run.execute(DUMMY, {"n": 2}, out, inputs={"fit": fit})
    assert DummySpec.model_validate(_json(out / "spec.json")).fit == fit.resolve()

    with pytest.raises(ConfigurationError, match="input"):
        run.execute(DUMMY, {"n": 2}, tmp_path / "other", inputs={"data": fit})


def test_list_analyses() -> None:
    analyses = run.list_analyses("examples.hctz_fitting.project:PROJECT")
    assert ("fit", "PK") in {(a["kind"], a["name"]) for a in analyses}


@pytest.mark.parametrize("project", ["examples.missing:PROJECT", "no_colon"])
def test_list_analyses_of_an_unknown_project(project: str) -> None:
    with pytest.raises(ConfigurationError):
        run.list_analyses(project)


def test_run_json_records_the_distribution_of_the_project(tmp_path: Path) -> None:
    """The project of a spec is resolved by the kind; the record only reads it."""
    out = tmp_path / "out"
    run.execute(DUMMY, {"n": 1, "project": "yaml.missing:PROJECT"}, out)
    record = _json(out / "run.json")
    assert record["versions"]["PyYAML"]
    assert record["git"] is None


def test_inputs_are_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    planned = run.plan(DUMMY, {"n": 2})
    run.compute_unit(DUMMY, planned, 0, tmp_path / "out", inputs={"fit": Path("fit")})
    stored = DummySpec.model_validate(_json(tmp_path / "out" / "spec.json"))
    assert stored.fit == (tmp_path / "fit").resolve()


def test_resume_with_the_inputs_at_another_path(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run.execute(DUMMY, {"n": 2, "seed": 1}, out, inputs={"fit": tmp_path / "a"})
    run.execute(
        DUMMY, {"n": 2, "seed": 1}, out, resume=True, inputs={"fit": tmp_path / "b"}
    )
    stored = DummySpec.model_validate(_json(out / "spec.json"))
    assert stored.fit == (tmp_path / "b").resolve()


def test_invalid_spec_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unknown"):
        run.execute(DUMMY, {"n": 1, "unknown": 2}, tmp_path / "out")
    with pytest.raises(ConfigurationError, match="n"):
        run.plan(DUMMY, {"n": "no number"})


def test_units_of_another_plan_are_refused(tmp_path: Path) -> None:
    first = run.plan(DUMMY, {"n": 10, "seed": 1})
    second = run.plan(DUMMY, {"n": 10, "seed": 2})
    run.compute_unit(DUMMY, first, 0, tmp_path, inputs={"fit": tmp_path / "a"})
    assert DummySpec.model_validate(_json(tmp_path / "spec.json")) == first.model_copy(
        update={"fit": (tmp_path / "a").resolve()}
    )
    # the same plan with its inputs at another path is the same plan
    run.compute_unit(DUMMY, first, 1, tmp_path, inputs={"fit": tmp_path / "b"})
    with pytest.raises(PlanMismatch, match="another planned spec"):
        run.compute_unit(DUMMY, second, 2, tmp_path)
    assert not (tmp_path / "parts" / "2.json").exists()
    with pytest.raises(PlanMismatch):
        run.merge(DUMMY, second, tmp_path)
