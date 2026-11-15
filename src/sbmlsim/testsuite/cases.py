"""The semantic cases of the SBML Test Suite on disk.

A case is a directory `cases/semantic/NNNNN/` with the model in several
encodings, `NNNNN-settings.txt` with the simulation and the tolerances,
`NNNNN-results.csv` with the expected results and `NNNNN-model.m` with the
description and the tags. `SemanticCase` reads the two metadata files, and
`SemanticSuite` is a directory of cases, downloaded from a commit of the
`develop` branch of the suite and cached.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import libsbml
import pandas as pd

from sbmlsim.atomic import write_text_atomic
from sbmlsim.model.model_sbml import to_level3
from sbmlsim.testsuite import cache

logger = logging.getLogger(__name__)

#: commit of the SBML Test Suite the tests, the documentation and the release
#: run against. The cases are taken from the branch `SUITE_BRANCH` and not from
#: a release, which are rare, so a case fixed upstream does not wait for the
#: next one (01511, #360); a scheduled job opens a pull request when the
#: semantic cases of the branch changed, which is where new failures are seen
SUITE_COMMIT = "bbcbc80c4daec79fea301299f981746408a1742b"

#: the branch of the suite the pinned commit follows
SUITE_BRANCH = "develop"

#: the repository of the suite on GitHub
SUITE_REPOSITORY = "sbmlteam/sbml-test-suite"

#: archive of a commit of the suite, `{commit}` is its hash
SUITE_URL = f"https://github.com/{SUITE_REPOSITORY}/archive/{{commit}}.zip"

#: the directory of the semantic cases in the repository of the suite
CASES_DIR = "cases/semantic"

#: the environment variable which points at the cases when they are not in
#: the cache
SUITE_PATH_VARIABLE = "SBMLSIM_TEST_SUITE_PATH"

#: the API of GitHub of the repository of the suite
API_URL = f"https://api.github.com/repos/{SUITE_REPOSITORY}"

#: SBML encodings of a case, the newest first: a case is run once, in the
#: newest encoding it provides. Reading the older encodings tests the reader,
#: not the simulation, and is covered by `sbmlutils`
ENCODINGS: tuple[str, ...] = (
    "l3v2",
    "l3v1",
    "l2v5",
    "l2v4",
    "l2v3",
    "l2v2",
    "l2v1",
    "l1v2",
)

#: the encoding every case is simulated in, the one the cases which do not ship
#: it are converted to
TARGET_ENCODING = "l3v2"


def converted_l3v2(
    path: Path, cid: str, source: Path, encoding: str
) -> tuple[Path, str]:
    """Get the case as SBML level 3 version 2, converting and caching it.

    The conversion is `to_level3`, written into the directory of the case as
    `<cid>-sbml-l3v2-converted.xml` with the sidecar
    `<cid>-sbml-l3v2-converted.json` which names the libsbml which wrote it and
    the encoding it was converted from. A cached pair is reused only when its
    sidecar names the running libsbml. The suite cache is shared by every
    process of a run, so both files are written atomically, the document first
    and the sidecar after it: the sidecar is what a reader trusts, so it never
    exists before the document it vouches for.

    Args:
        path: directory of the case, which the conversion is written into.
        cid: identifier of the case.
        source: the newest encoding the suite ships for the case.
        encoding: the encoding of `source`, e.g. `l2v5`, recorded in the sidecar.

    Returns:
        The path of the converted document and the error message, which is empty
        unless libsbml refused to convert it, in which case nothing is written.

    Raises:
        OSError: If the directory of the case cannot be written to, naming the
            case and the directory. There is no fallback to an uncached
            conversion: a suite directory which is read only is a setup to fix
            (copy it or set `SBMLSIM_TEST_SUITE_PATH`), not one to hide.
    """
    converted = path / f"{cid}-sbml-{TARGET_ENCODING}-converted.xml"
    sidecar = path / f"{cid}-sbml-{TARGET_ENCODING}-converted.json"
    version = libsbml.getLibSBMLDottedVersion()
    try:
        if (
            converted.exists()
            and json.loads(sidecar.read_text(encoding="utf-8"))["libsbml"] == version
        ):
            return converted, ""
    except OSError, ValueError, KeyError, TypeError:
        logger.debug("'%s': the cached conversion is unusable", cid)

    try:
        text = to_level3(source.read_text(encoding="utf-8"), cid)
    except ValueError as err:
        logger.debug("'%s': the case could not be converted: %s", cid, err)
        return converted, str(err)
    try:
        write_text_atomic(converted, text)
        write_text_atomic(sidecar, json.dumps({"libsbml": version, "from": encoding}))
    except OSError as err:
        raise OSError(
            f"'{cid}': the conversion cannot be written into '{path}': {err}"
        ) from err
    return converted, ""


#: keys of `NNNNN-model.m` which are read, all of them single line
_TAG_PATTERN = re.compile(
    r"^(componentTags|testTags|testType|generatedBy|packagesPresent):(.*)$",
    re.MULTILINE,
)


def _split(value: str) -> list[str]:
    """Split a comma separated value of a metadata file.

    Args:
        value: the text behind the key, may be empty.

    Returns:
        The entries without surrounding whitespace, empty for an empty value.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class SemanticCase:
    """A semantic case of the SBML Test Suite.

    Attributes:
        cid: identifier of the case, i.e. its directory name such as `00028`.
        path: directory of the case.
        model_path: the model which is simulated, level 3 version 2: the file
            of the suite, or its conversion when the suite has none.
        start: start time of the simulation.
        duration: end time of the simulation.
        steps: number of steps, so the results have `steps + 1` points.
        variables: the columns the results are compared on, without `time`.
        amount: the variables which are compared as an amount.
        concentration: the variables which are compared as a concentration.
        absolute_tolerance: absolute tolerance of the comparison.
        relative_tolerance: relative tolerance of the comparison.
        component_tags: SBML components the case uses, e.g. `EventWithDelay`.
        test_tags: what the case tests, e.g. `NonConstantParameter`.
        test_type: kind of the case, `TimeCourse` for the cases which are run.
        generated_by: `Analytic` or `Numeric`, how the results were produced.
        packages: SBML packages the case needs, e.g. `fbc`.
        converted_from: the encoding a converted case came from, e.g. `l2v5`,
            empty when the suite ships `l3v2`.
        conversion_error: why libsbml refused to convert the case, else empty.
    """

    cid: str
    path: Path
    model_path: Path
    start: float
    duration: float
    steps: int
    variables: list[str]
    amount: list[str]
    concentration: list[str]
    absolute_tolerance: float
    relative_tolerance: float
    component_tags: frozenset[str] = field(default_factory=frozenset)
    test_tags: frozenset[str] = field(default_factory=frozenset)
    test_type: str = ""
    generated_by: str = ""
    packages: frozenset[str] = field(default_factory=frozenset)
    converted_from: str = ""
    conversion_error: str = ""

    @staticmethod
    def settings(path: Path) -> dict[str, str]:
        """Read the settings file of a case.

        Args:
            path: `NNNNN-settings.txt` of the case.

        Returns:
            The keys of the file with their raw values, values may be empty.
        """
        settings: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if _:
                settings[key.strip()] = value.strip()
        return settings

    @staticmethod
    def tags(path: Path) -> dict[str, str]:
        """Read the tags of the description of a case.

        The description is prose with a header of `key: value` lines; only the
        single line keys of the header are read, i.e. not the synopsis.

        Args:
            path: `NNNNN-model.m` of the case.

        Returns:
            The tag keys with their raw values.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        return {key: value.strip() for key, value in _TAG_PATTERN.findall(text)}

    @classmethod
    def from_directory(cls, path: Path) -> SemanticCase | None:
        """Read a case from its directory.

        A case which is not a timecourse simulation is not read: the flux
        balance cases have no duration and the stochastic cases are not part
        of the semantic suite.

        Args:
            path: directory of the case, named after its identifier.

        Returns:
            The case, or `None` if it is not a timecourse case or has no
            model in a known encoding.
        """
        cid = path.name
        settings_path = path / f"{cid}-settings.txt"
        if not settings_path.exists():
            logger.debug("'%s': no settings file, the case is skipped", cid)
            return None

        settings = cls.settings(settings_path)
        # a case without a duration is not a timecourse, i.e. it is a flux
        # balance case, which needs a steady state and not a simulation
        if not settings.get("duration"):
            return None

        model = cls.model_of(path, cid)
        if model is None:
            logger.debug("'%s': no model in a known encoding", cid)
            return None
        model_path, converted_from, conversion_error = model

        model_file = path / f"{cid}-model.m"
        tags = cls.tags(model_file) if model_file.exists() else {}

        return cls(
            cid=cid,
            path=path,
            model_path=model_path,
            start=float(settings.get("start") or 0.0),
            duration=float(settings["duration"]),
            steps=int(settings.get("steps") or 0),
            variables=_split(settings.get("variables", "")),
            amount=_split(settings.get("amount", "")),
            concentration=_split(settings.get("concentration", "")),
            absolute_tolerance=float(settings.get("absolute") or 1e-9),
            relative_tolerance=float(settings.get("relative") or 1e-6),
            component_tags=frozenset(_split(tags.get("componentTags", ""))),
            test_tags=frozenset(_split(tags.get("testTags", ""))),
            test_type=tags.get("testType", ""),
            generated_by=tags.get("generatedBy", ""),
            packages=frozenset(_split(tags.get("packagesPresent", ""))),
            converted_from=converted_from,
            conversion_error=conversion_error,
        )

    @staticmethod
    def model_of(path: Path, cid: str) -> tuple[Path, str, str] | None:
        """Get the model of a case as level 3 version 2.

        The file of the suite when it ships `l3v2`, which is never rewritten,
        else the conversion of the newest encoding it provides.

        Args:
            path: directory of the case.
            cid: identifier of the case.

        Returns:
            The SBML file, the encoding it was converted from (empty when the
            suite ships `l3v2`) and the conversion error (empty unless libsbml
            refused, the file is then the newest encoding as the suite ships
            it), or `None` if the case has no model.
        """
        for encoding in ENCODINGS:
            model_path = path / f"{cid}-sbml-{encoding}.xml"
            if not model_path.exists():
                continue
            if encoding == TARGET_ENCODING:
                return model_path, "", ""
            converted, error = converted_l3v2(path, cid, model_path, encoding)
            if error:
                return model_path, encoding, error
            return converted, encoding, ""
        return None

    @property
    def encoding(self) -> str:
        """Get the SBML encoding which is simulated, e.g. `l3v2`."""
        if self.converted_from and not self.conversion_error:
            return TARGET_ENCODING
        match = re.search(r"-sbml-(l\d+v\d+)\.xml$", self.model_path.name)
        return match.group(1) if match else ""

    @property
    def selections(self) -> list[str]:
        """Get the roadrunner selections of the compared variables.

        A variable of the `concentration` list is the concentration of a
        species, i.e. `[S1]`, everything else is read under its own
        identifier. `time` is always selected first.
        """
        concentration = set(self.concentration)
        return [
            "time",
            *(f"[{v}]" if v in concentration else v for v in self.variables),
        ]

    def expected(self) -> pd.DataFrame:
        """Read the results a correct simulator produces.

        The first column of the results is the time and the ones after it are
        the variables of the settings, in their order; the columns are named
        from that rather than from the header. The header spells the time
        `time` in most cases and `Time` in others, and a model may carry a
        parameter of either name: case 01820 has the parameters `time`, `Time`
        and `TIME` and a header of `Time,time,Time,TIME`, which neither a case
        insensitive rename nor `read_csv` alone can tell apart.

        Returns:
            The expected results with a `time` column and one column per
            variable of the case.

        Raises:
            ValueError: If the results do not carry one column per variable.
        """
        df = pd.read_csv(self.path / f"{self.cid}-results.csv")
        if len(df.columns) != len(self.variables) + 1:
            raise ValueError(
                f"Case '{self.cid}' compares {len(self.variables)} variables "
                f"and its results have {len(df.columns)} columns, which is "
                f"not the time and one column per variable"
            )
        df.columns = ["time", *self.variables]
        return df


@dataclass(frozen=True)
class SemanticSuite:
    """The semantic cases of a commit of the SBML Test Suite.

    Attributes:
        path: directory which holds the case directories.
        commit: commit of the suite the cases are from.
    """

    path: Path
    commit: str

    def __len__(self) -> int:
        """Get the number of timecourse cases of the suite."""
        return len(list(self.cases()))

    def cases(self) -> Iterator[SemanticCase]:
        """Iterate the timecourse cases of the suite, by identifier.

        Yields:
            Every case which is a timecourse simulation, in the order of the
            case identifiers.
        """
        for directory in sorted(self.path.iterdir()):
            if not directory.is_dir():
                continue
            case = SemanticCase.from_directory(directory)
            if case is not None:
                yield case

    def case(self, cid: str) -> SemanticCase:
        """Get a single case by its identifier.

        Args:
            cid: identifier of the case, e.g. `00028`.

        Returns:
            The case.

        Raises:
            ValueError: if the suite has no such timecourse case.
        """
        case = SemanticCase.from_directory(self.path / cid)
        if case is None:
            raise ValueError(f"'{cid}': no timecourse case in the suite '{self.path}'")
        return case

    # ------------------------------------------------------------------
    # the commit of the suite
    # ------------------------------------------------------------------
    @staticmethod
    def cache_path(commit: str) -> Path:
        """Get the directory a commit of the suite is unpacked into.

        `SBMLSIM_TEST_SUITE_PATH` overrides it, e.g. for an offline machine
        which has the cases somewhere else. Otherwise it is
        `sbmlsim/test-suite/<commit>/semantic` in the user cache, i.e. in
        `XDG_CACHE_HOME` or `~/.cache`.

        Args:
            commit: commit of the suite.

        Returns:
            The directory the cases of the commit live in.
        """
        return cache.cache_path(SUITE_PATH_VARIABLE, "test-suite", commit, "semantic")

    @staticmethod
    def latest_commit(branch: str = SUITE_BRANCH) -> str:
        """Get the newest commit of a branch of the suite from GitHub.

        Args:
            branch: the branch of the suite.

        Returns:
            The hash of the head of the branch.
        """
        commit: str = _github(f"commits/{urllib.parse.quote(branch)}")["sha"]
        return commit

    @staticmethod
    def cases_changed(pinned: str, commit: str) -> bool:
        """Check whether the semantic cases of a commit differ from a pinned one.

        The last commit before `commit` which changed the semantic cases is
        compared with `pinned`: when `pinned` contains it, the cases of both
        commits are the same, e.g. after a commit which changed only the
        test runner of the suite.

        Args:
            pinned: the pinned commit.
            commit: the commit which may be adopted, e.g. the head of the
                branch, see `latest_commit`.

        Returns:
            Whether `commit` has cases `pinned` has not.
        """
        query = urllib.parse.urlencode(
            {"sha": commit, "path": CASES_DIR, "per_page": 1}
        )
        changes = _github(f"commits?{query}")
        if not changes:
            return False
        last = changes[0]["sha"]
        # `identical` or `ahead`: the pinned commit is the last change of the
        # cases or a descendant of it
        status = _github(f"compare/{last}...{pinned}")["status"]
        return status not in ("identical", "ahead")

    @classmethod
    def cached(cls, commit: str = SUITE_COMMIT) -> SemanticSuite | None:
        """Get a commit of the suite if it is already on this machine.

        Args:
            commit: commit of the suite.

        Returns:
            The suite, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(commit)
        return cls(path=path, commit=commit) if path.is_dir() else None

    @classmethod
    def load(cls, commit: str = SUITE_COMMIT) -> SemanticSuite:
        """Get a commit of the suite, downloading it if it is not cached.

        The archive of the commit is the whole repository, about 100 MB, of
        which only the semantic cases are unpacked.

        Args:
            commit: commit of the suite.

        Returns:
            The suite with its cases unpacked in the cache.

        Raises:
            OSError: if the commit cannot be downloaded.
        """
        suite = cls.cached(commit)
        if suite is not None:
            # what a fetch which was killed left next to the cache, which an
            # override is not
            if not cache.is_overridden(SUITE_PATH_VARIABLE):
                cache.remove_stale(suite.path)
            return suite

        path = cls.cache_path(commit)
        url = SUITE_URL.format(commit=commit)
        logger.info("Downloading the SBML Test Suite '%s' from '%s'", commit, url)
        cache.fetch(url, path, select=cls._cases_dir, members=_is_case_member)
        return cls(path=path, commit=commit)

    @staticmethod
    def _cases_dir(staging: Path) -> Path:
        """Get the directory of the case directories of an unpacked archive.

        The archive of a commit holds the repository in a directory named
        after the commit, with the cases in `cases/semantic`.

        Args:
            staging: directory the archive was unpacked into.

        Returns:
            The directory which holds the case directories.

        Raises:
            OSError: if it holds no cases.
        """
        for candidate in sorted(staging.glob(f"*/{CASES_DIR}")):
            if any(p.is_dir() and p.name.isdigit() for p in candidate.iterdir()):
                return candidate
        raise OSError(f"No case directories in the unpacked suite '{staging}'")


def _is_case_member(name: str) -> bool:
    """Check whether a member of the archive of a commit is part of the cases.

    The archive is the whole repository, whose test runner and web interface
    are more than three times the size of the cases.

    Args:
        name: the name of the member, e.g.
            `sbml-test-suite-<commit>/cases/semantic/00001/00001-settings.txt`.

    Returns:
        Whether the member is below the semantic cases.
    """
    _, _, below = name.partition("/")
    return below.startswith(f"{CASES_DIR}/")


def _github(path: str) -> Any:
    """Get a resource of the API of GitHub of the repository of the suite.

    The request is authenticated with the token in `GITHUB_TOKEN` or
    `GH_TOKEN` when one is set: GitHub allows 60 anonymous requests per hour
    and address, which the shared addresses of the runners of GitHub Actions
    exhaust (`HTTP Error 403: rate limit exceeded`).

    Args:
        path: the path below the repository, e.g. `commits/develop`.

    Returns:
        The JSON of the resource.
    """
    request = urllib.request.Request(
        f"{API_URL}/{path}", headers={"Accept": "application/vnd.github+json"}
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)
