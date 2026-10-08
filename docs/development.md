# Development

Contributions are welcome. The repository is [matthiaskoenig/sbmlsim](https://github.com/matthiaskoenig/sbmlsim); development happens against the `develop` branch via pull requests.

## Branch model

Two branches are permanent:

- **`develop`** is the default branch and the branch everything is integrated into. The documentation on [matthiaskoenig.github.io/sbmlsim](https://matthiaskoenig.github.io/sbmlsim) is published from it.
- **`main`** tracks the latest published release. It is fast-forwarded to the released commit by the `sync-main` job of the `CI-CD` workflow after the package went to pypi, so `main` and the newest version on pypi always agree. Nothing is developed on `main` and nothing is merged into it by hand.

Work happens on short lived branches off `develop`, which GitHub deletes after the merge. Releases are tagged on `develop`, see [Release](#release).

## Pull requests

Neither branch accepts a direct push, every change goes through a pull request against `develop`. This includes the maintainer, there is no bypass.

A pull request can only be merged once the four required checks are green:

| check   | workflow      | content                                                              |
| ------- | ------------- | -------------------------------------------------------------------- |
| `tests` | `ci-cd.yml`   | the test matrix, linux with python 3.13 and 3.14, macos and windows with 3.14 |
| `ruff`  | `lint.yml`    | `ruff check` and `ruff format --check`                                |
| `ty`    | `lint.yml`    | `ty check`                                                            |
| `docs`  | `docs.yml`    | the zensical build including the api reference and the agent files    |

`tests` aggregates the test matrix into a single job, so the name of the required check stays the same when the matrix changes.

Every job sets up its environment with the action `.github/actions/setup`: uv with its cache, the python of the job, the libpython roadrunner needs on linux, and `uv sync` with the extras of the job. The jobs run the same commands as a local environment (`pytest`, `ty check`, `zensical build`), the tests against the package installed as a wheel (`uv sync --no-editable`). A run of a pull request is cancelled by the next push to it; a push to `develop` or `main` is never cancelled, a release waits for the run of its commit.

Further rules of a pull request:

- conversations have to be resolved before the merge
- an approval is dismissed when new commits are pushed
- the history stays linear, i.e., a pull request is merged with squash or rebase; merge commits are disabled
- the maintainer is the code owner of the repository (`.github/CODEOWNERS`) and is requested for review on every pull request. A pull request of a contributor is therefore reviewed and merged by the maintainer, who has the only write access. The rulesets themselves do not require an approval: on a personal repository a ruleset cannot ask for an approval only from somebody else, and requiring one would block the pull requests of the maintainer, who cannot approve their own. Once a second person has write access, a ruleset requiring an approving review of a code owner can be added

[Auto-merge](https://docs.github.com/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request) is enabled for the repository, so a pull request can be queued and is merged as soon as the checks pass and the required approval is there.

### Repository policies { #repository-policies }

The protection is implemented with [repository rulesets](https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets). They are part of the repository in `.github/rulesets/` instead of only living in the web interface, so a change to a policy is reviewed like any other change:

| ruleset                 | applies to | rules                                                                                                                                       |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `develop.json`          | `develop`  | pull request required, the four checks above, resolved conversations, linear history, no force push, no deletion. **No bypass, for anybody.** |
| `main.json`             | `main`     | no force push, no deletion, no bypass. The fast-forward of the release workflow needs none, only a force push would be rejected. `main` mirrors `develop`, whose history carries merge commits from before merge commits were disabled, so `main` cannot require a linear history |
| `tags.json`             | all tags   | a tag cannot be deleted or moved, so a release tag keeps pointing at what was released                                                       |

Changing a policy means changing the json and applying it:

```bash
.github/rulesets/apply.sh
```

The script is idempotent: it updates the rulesets which exist and creates the missing ones. It also sets the merge settings of the repository, i.e., auto-merge, delete branch on merge, and squash and rebase as the only merge methods. It needs the [github cli](https://cli.github.com) authenticated as a user with admin permission on the repository.

## Setup development environment

Development needs [uv](https://docs.astral.sh/uv/) and a checkout of the repository:

```bash
git clone https://github.com/matthiaskoenig/sbmlsim.git
cd sbmlsim
```

A single sync creates the virtual environment in `.venv`, installs `sbmlsim` into it in editable mode and adds the complete tooling:

```bash
uv sync --extra dev
```

The `dev` extra contains everything used below, i.e., pytest, ruff, ty, tox with tox-uv, pre-commit, zensical and bump-my-version, so nothing has to be installed separately. The python version is taken from `.python-version` (currently 3.14); to work against the oldest supported version instead use `uv sync --extra dev --python 3.13`, which replaces the environment.

The tools are then run either with `uv run <command>`, which uses the environment without activating it, or from the activated environment:

```bash
source .venv/bin/activate        # Linux and macOS
.venv\Scripts\activate           # Windows
```

The commands in this document are written without the `uv run` prefix; prepend it if the environment is not activated.

The last step installs the git hook:

```bash
uv run pre-commit install          # install the hook, once per checkout
uv run pre-commit run --all-files  # check the current state of the repository
```

From now on every commit is checked with ruff (lint and format) and ty, i.e., the same checks that run in continuous integration. On a commit only the changed files are looked at, `--all-files` checks the whole repository and is what a newly added hook should be tried with.

## Testing

The tests are written with pytest, tox runs them against every supported python version.

The tox environments are named after the interpreter (`py3.13` to `py3.14`, see `envlist` in `tox.ini`), a single one is run with

```bash
tox r -e py3.14
```

and the complete matrix in parallel with

```bash
tox run-parallel
```

This needs the interpreters to be available, which uv installs with `uv python install 3.13 3.14`. Continuous integration does not use tox, it runs `pytest` in an environment of the python of the job, see [pull requests](#pull-requests).

To run the tests directly against the development environment use

```bash
pytest                                          # the full suite
pytest -n 0                                     # in one process, e.g. for --pdb
pytest tests/simulation/test_simulation.py                  # a single module
pytest tests/simulation/test_simulation.py::test_timecourse  # a single test
```

The tests run in parallel, `-n auto --dist worksteal` in the `addopts` of `pyproject.toml` gives pytest-xdist one worker per core, and a worker which is idle takes over the tests which wait at a busy one; `-n 0` on the command line runs everything in one process, which the debugger needs.

The `conftest.py` at the root of the repository selects the non-interactive matplotlib backend for the session and puts the repository on `sys.path`, so that the tests can import the examples.

Some tests are skipped on purpose: the simulation experiment examples are marked with `pytest.mark.skip` while that part of the package is reworked. The skips are listed with `pytest -rs`.

`tests/examples/test_example_scripts.py` runs the examples as `python -m examples.<module>` in a temporary working directory, so a broken example fails the test suite.

## Linting and formatting

Linting and formatting use [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check     # lint
ruff format    # format
```

The docstring rules (`D`) are enforced for the package, not for `examples/` and `tests/`, which are scripts and fixtures; the model definitions of the examples import the names of `sbmlutils.factory` with a star import, so `F403`/`F405` are ignored there as well, see `.ruff.toml`.

## Type checking

Type checking is performed with [ty](https://docs.astral.sh/ty/):

```bash
uv run ty check
```

ty resolves the imports in the environment of the project, which `uv sync --extra dev` creates; the `ty` check of continuous integration runs the same command and the pre-commit hook runs it on every commit.

The configuration lives in `[tool.ty]` in `pyproject.toml`. Warnings are treated as errors, so the codebase is kept free of diagnostics. Suppress an unavoidable diagnostic with a rule specific `# ty: ignore[rule-name]` rather than a blanket comment.

libsbml and roadrunner have no type stubs and create their objects through a SWIG layer, so ty sees an untyped API. Annotate their objects (`doc: libsbml.SBMLDocument = ...`) and use the explicit getters (`getVariable()`) rather than the attributes the SWIG layer synthesizes (`variable`), which the type checker cannot see.

## Examples

The examples are runnable scripts in `examples/`, they are not part of the package. They are run as modules from the root of the repository:

```bash
python -m examples.timecourse
python -m examples.demo.demo
```

An example writes what it creates into the current working directory and never opens a window: a plotting example saves its figure to a file. `tests/examples/test_example_scripts.py` runs the example scripts in a temporary directory, so a broken example fails the test suite. See `examples/README.md`.

## Documentation

The documentation is built with [Zensical](https://zensical.org/), the static site generator of the Material for MkDocs authors. The sources are markdown files in `docs/`, the site is configured in `zensical.toml` in the repository root. Nothing rendered is committed: the site is built by the `documentation` workflow on every push and published to [matthiaskoenig.github.io/sbmlsim](https://matthiaskoenig.github.io/sbmlsim) from the `develop` branch.

Build the site into `site/`:

```bash
uv run zensical build --clean
```

For writing, the preview rebuilds on save:

```bash
uv run zensical serve
```

The API reference is rendered from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/); a page in `docs/api/` only contains the module directive:

```markdown
# simulation.timecourse

::: sbmlsim.simulation.timecourse
```

Docstrings are therefore the place to document functions and classes, the markdown files provide the narrative around them. Adding a module to the reference means adding such a page and an entry to `nav` in `zensical.toml`.

### Files for agents { #files-for-agents }

Agents and language models read markdown, not rendered html. `scripts/llms_txt.py` writes the files of the [llms.txt convention](https://llmstxt.org/) into the built site, i.e., [llms.txt](https://matthiaskoenig.github.io/sbmlsim/llms.txt) as an annotated index of all pages, [llms-full.txt](https://matthiaskoenig.github.io/sbmlsim/llms-full.txt) with the complete documentation in a single file, and the markdown of every page next to its html (`/creation.md` for `/creation/`). The markdown of the API reference is generated from the docstrings with `inspect`, since the pages themselves only contain the mkdocstrings directive.

```bash
uv run zensical build --clean
uv run python scripts/llms_txt.py
```

The `documentation` workflow runs both steps, so the files are regenerated with every push. `docs/robots.txt` points crawlers at the sitemap and at these files. Zensical will provide agent context files itself at some point, then this script can go.

## Release

A release is made from `develop`. Since `develop` only accepts pull requests, the release is prepared on a branch and tagged once that pull request is merged:

1. branch off `develop`: `git switch -c release/x.y.z develop`
2. write the release notes for the version in `release-notes/x.y.z.md`
3. make sure everything passes: `tox run-parallel`, `ruff check`, `uv run ty check`
4. check the version bump: `uvx bump-my-version bump [major|minor|patch] --dry-run -vv`
5. bump the version: `uvx bump-my-version bump [major|minor|patch]`, which updates `src/sbmlsim/__init__.py` and `CITATION.cff` and commits. It does not create the tag; a squash or rebase merge would rewrite the commit and leave the tag behind on a commit which is not part of `develop`
6. push the branch, open the pull request against `develop` and merge it once the checks are green
7. tag the merged commit on `develop` and push the tag:

    ```bash
    git switch develop
    git pull
    git tag x.y.z
    git push origin x.y.z
    ```

    This starts the `CI-CD` workflow, which publishes to [pypi](https://pypi.org/project/sbmlsim/), creates the GitHub release from `release-notes/x.y.z.md` and fast-forwards `main` to the tagged commit. It does not run the test matrix again: the merge into `develop` ran it on the same commit, and the job `tested on develop` waits for that run and stops the release unless it passed, so the tag can be pushed right after the merge. Next to it the SBML Test Suite and the PEtab SciML test suite run against their pins and the submission to the SBML Test Suite Database is written, which the release attaches; a release takes a few minutes once the run on `develop` has finished. Check the version before pushing, a tag cannot be moved or deleted afterwards.

8. test the installation from pypi in a fresh environment:

    ```bash
    uv venv --python 3.14
    uv pip install sbmlsim
    ```

9. once Zenodo has archived the release, update the citation information, i.e., `date-released` in `CITATION.cff` and the version, date and version DOI of the release in the citation of `README.md` and `docs/index.md`. `bump-my-version` only updates the version, not the date and the DOI, which are only known after the release. These changes go in through a pull request like everything else
