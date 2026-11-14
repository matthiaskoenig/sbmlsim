# Release of a model repository

Every model of a package can be published as a repository of its own (`hctz-model`, `glimepiride-model`, ...), archived by Zenodo and cited by its DOI. `sbmlsim.release` makes the complete release of such a repository from the [project](project.md) of the model package, the same way for every model: the build is the kind `release` of the [run layer](run.md) and the publishing the local step `python -m sbmlsim.release publish`.

The release repository holds only what the build generates. The models, the simulations and the fit reports are created in the release repository from the exported package and its released dependencies, so a release proves that the model runs on its own. Nothing in a release repository is edited by hand: a change is made in the model package and released again. Once a model has its release metadata and the one-time setup is done, a release needs no manual step before, during or after the commands.

## The release metadata of a project

What the package does not know about a release is `Project.release`, the `ReleaseSettings` of `sbmlsim.release.settings` (standard library only, a project which is not released has `None`):

```python
from sbmlsim.release.settings import Author, ReleaseSettings

release = ReleaseSettings(
    title="Model of hydrochlorothiazide",
    repository="matthiaskoenig/hctz-model",
    doi="10.5281/zenodo.15427418",
    authors=[
        Author(
            "König, Matthias",
            orcid="0000-0003-1725-179X",
            affiliation="Humboldt-University Berlin, Institute for Biology",
        ),
    ],
    keywords=["hydrochlorothiazide", "pharmacokinetics", "SBML"],
    docker_image="matthiaskoenig/hctz",
    funding="The funding of the work on the model.",
    description="A model of hydrochlorothiazide.",
    first_year=2025,
)
print(release.site_url)
print(release.missing())
```

```text title="Output"
https://matthiaskoenig.github.io/hctz-model/
[]
```

- `title`: the title of the release: the Zenodo record, the citation, the heading of the README and the name of the docs site.
- `repository`: the GitHub repository as `owner/name`; the docs site is `https://<owner>.github.io/<name>/`.
- `doi`: the concept DOI of Zenodo, which cites every version.
- `authors`: the authors of the citation in order, each with the name as `"Family, Given"`, the ORCID and the affiliation.
- `citation(year=None, markdown=True)`: the citation of the release, the one formatter of it: the authors as `"Family, G."`, the last joined with `&`, the year, the title, Zenodo and the DOI, e.g. `König, M. (2026). *A model of hydrochlorothiazide*. Zenodo. https://doi.org/10.5281/zenodo.15427418`. The README and the docs of the release carry it with the year of the build, and the app of a model shows it in its footer, `App(citation=RELEASE.citation(markdown=False))`, the plain text without the emphasis of the title (see [Interactive panels](serve.md#the-app-of-a-model)).
- `keywords`, `funding`, `description`: the Zenodo record (`.zenodo.json`), the docs and the README. `title`, `repository`, `doi`, `authors`, `keywords`, `docker_image`, `funding`, `description` and `first_year` are required; `missing()` lists the ones which are empty.
- `docker_image`: the image which is pushed as `<docker_image>:<version>` and `:latest`. The image of the version is built and pushed while the registry lacks it, a pushed version is never rebuilt and `:latest` is then tagged on the registry from it.
- `exclude`: paths of the model folder which are not released, by default `TODO.md`, `literature` and `thesis`. An empty `exclude` releases the whole model folder.
- `first_year`: the year of the first release, the start of the copyright.
- `framework`: `sbmlsim` by default, the package the release is built with; the lock of the release must contain it, `release.json` records its version and the release notes say "with sbmlsim <version>". A framework on top of sbmlsim names itself.
- `core_libraries`: `sbmlsim` and `sbmlutils` by default, whose release notes the release notes summarize; `library_owner`: `matthiaskoenig`, the owner of their repositories.
- `archive`: the COMBINE archive the factory writes, a path relative to the root of the project; the build copies it into the release, the models page links it and the GitHub release attaches it, which a release without one is not published with (see [Licenses](licenses.md) for its contents).
- `hook`: the release hook of a framework, see [Frameworks](#frameworks).

A project without `release`, with an empty field or a `first_year` which is no positive year cannot be released; the command names the missing fields (`missing()`).

## One-time setup

Once per machine:

- `docker login` with an account which may push the `docker_image`; the buildx plugin of docker (part of Docker Desktop and Docker Engine) tags `:latest` on the registry.
- `gh auth login` with an account which may push to the repository and create its releases.

Once per release repository:

- The repository exists on GitHub (the command does not create it), it may be empty; the command clones it when there is no checkout.
- The GitHub integration of Zenodo is switched on for the repository (Zenodo, account settings, GitHub), so that Zenodo archives every GitHub release under the concept DOI.

A release depends on released packages only: the versions of sbmlsim and of the libraries which the lock of the release holds must be on PyPI or on a package index of the lock of the source. A package of such an index, e.g. libroadrunner of the builds of its develop branch ([Installation](installation.md)), is taken from it in the release as well: the release project declares the index as an explicit index, pins the package to it (`tool.uv.sources`) and has it as a dependency, so its lock has the build of the source lock.

## The build

The kind `release` builds the release into the out: it clones the release repository into `out/repository` (from GitHub, or from a checkout given as `repository`, which is only read and only a cache: the clone takes `main` of the release repository, and the clone of an earlier run into the out is fast-forwarded to it; a checkout whose `main` has commits which are not pushed is refused), checks that the source repository of the project is clean and that the version is greater than the last release, exports the tracked files of the package into `src/`, writes the project files (`pyproject.toml` locked to the versions of the source, the Dockerfile, the workflows, `.zenodo.json`), runs the analyses of the release in its own environment (`uv lock`, `uv sync` and then `python -m sbmlsim.run` with the specs it writes into `specs/`: `factory` first, whose models replace the ones of the exported package, so every analysis and the released models agree, then `simulate`, `simulate` over the draws of the adopted fits when the default variant has their distributions, `report` of the adopted fits and `credibility` of a project with a declaration), copies their results into `models/`, `figures/`, `simulations/` and `simulations_uncertainty/` (with the static figures in `_figures`), `fit/<fit id>/` and `credibility/`, writes the release notes (the changes of the core libraries, the libraries, the data, the models, the experiments and the adopted fits since the last release), the pages of the site and the README, removes every file no stage wrote and writes the manifest `release.json` with a hash of every file. It commits nothing:

```yaml title="release.yaml"
project: my_model:PROJECT
version: 0.8.0
```

```bash
python -m sbmlsim.run release release.yaml --out results/release
python -m sbmlsim.release build my_model:PROJECT --version 0.8.0 --out results/release
python -m sbmlsim.release publish my_model:PROJECT --version 0.8.0 --out results/release
```

The steps of the build, in order; the first failure stops it:

1. **Checks**, before anything is built: the version, the release metadata, the source repository is clean and, for a run which publishes, docker with its buildx plugin and `gh`.
2. **Build**: the release repository is cloned into `out/repository` and checked (on `main`, up to date, without local commits and without changes which no release made, the version greater than the last tag, every adopted fit with its record), then the export of the package, the project files, the analyses of the run layer, the release notes, the docs and the README, the prune and the manifest `release.json`. The figures of the model are shown by the README and the site: the overview `figures/<name>_model.png` (or `.svg`) of the folder `figures` beside the factory output, and the diagrams which the factory draws beside its models, every SVG of `models` (e.g. the schematic `<stem>.svg` and the anatomy `<stem>_body.svg` of a model of the PBPK library of pkpdsim); the models page shows a diagram at the model whose stem is its name or the start of its name followed by `_`. A build of the same spec in the output is built again, so a failed or interrupted build is finished by the next run (`--resume`).

`python -m sbmlsim.release build` takes the project and `--version` and `--out` of the kind and the options `--repository PATH` (a checkout of the release repository to clone from; the clone of the output is fast-forwarded to `origin/main`, one on another branch or with commits of its own is refused), `--framework-path PATH` (a local checkout of the framework instead of the released one, for a test of a framework which is not released yet; such a build is never published), `--workers N` (the simulations, the CPUs of the quota without) and `--resume`. Its exit codes are those of `python -m sbmlsim.run`: 0 done, 2 a bad command line, 3 the build failed and 4 a configuration error (a refused release), its message on stderr.

## Review, then publish

`publish` is the local step which never runs as a kind. Run it on the output of a build which was reviewed: it checks that the build is still the build of the current source (`matches`: the same version, the same commit of the source, the same framework, every file as it was built; a file changed during the review makes it refuse and the build must be run again), that docker, its buildx plugin and `gh` are ready and that the lock has released packages only, and then builds the Docker image, commits, tags, sets up GitHub Pages, pushes, pushes the image and creates the GitHub release with the release notes and the COMBINE archive. Then a clean checkout of the release repository on `main` is fast-forwarded to the release; one with changes or on another branch is left as it is with a warning.

A publishing which failed half way is finished by running the same command again: every step checks whether it is done already (the commit, the tag, the push, the image, the GitHub release) and is skipped then. If the last commit of the release repository of the output is the release of the version (the tag is missing or at that commit), the build is skipped and the publishing steps resume, even if the source moved on since. The exit codes of `publish`: 0 published, 1 refused or failed, 2 a bad command line and 4 a project which is not loaded.

## Frameworks

A framework on top of sbmlsim names itself and its core libraries in the release settings and contributes the content of its domain through a `ReleaseHook` (`sbmlsim.release.hooks`, `ReleaseSettings.hook`): templates which replace or add to the templates of the release, pages of the site, values of the pages, the data of the page of the credibility (e.g. the evidence of its extensions, shown by the parts `docs/credibility-acceptance.md.j2`, `docs/credibility-uncertainty.md.j2`, `docs/credibility-context.md.j2` and `docs/credibility-sections.md.j2` of its templates), the study a file of the data belongs to (`data_study(data_dir, path)`, by default its folder below the package, `<data dir>/<folder>`) and sections of the release notes. The owner of the GitHub repositories of the core libraries, whose release notes are summarized, is `library_owner`.

A framework registers the hook in the release settings of its projects (`ReleaseSettings.hook`, an object or a reference `module:attribute`; `Project.release_hook()` resolves it), see `sbmlsim.release.hooks`.

## The COMBINE archive

`sbmlsim.release.archive.write_archive(models, out, name=..., copyright=...)` writes the COMBINE archive of the models of a release for a factory: the flat model of a model with submodels, the differential equations of every flat model as markdown and its antimony, the license texts and `<name>.omex` with the README and further models and files.

## simrun

The kind `release` is the task type `sbmlsim.release` of simrun, an unsplit task which builds into its output port, so a workflow can build a release on a cluster; publishing stays a local step, see [simrun](run.md#simrun).
