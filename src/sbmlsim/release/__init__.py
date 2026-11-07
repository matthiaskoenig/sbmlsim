"""The release of a model package as a repository of its own.

The kind `release` of the run layer (`sbmlsim.run.release_kinds`) builds the
release of a project into an output, `python -m sbmlsim.release publish`
publishes it (`sbmlsim.release.workflow`). `sbmlsim.release.settings` holds the
release metadata of a project (`sbmlsim.project.Project.release`) and
`sbmlsim.release.hooks` what a framework adds to a release; both use the
standard library only, so a model package which defines its project imports
them cheaply. `sbmlsim.release.archive` writes the COMBINE archive of the
models of a release.

`import sbmlsim` does not import this package.
"""

__all__: list[str] = []
