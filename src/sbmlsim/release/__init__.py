"""The release of a model package as a repository of its own.

`sbmlsim.release.settings` holds the release metadata of a project
(`sbmlsim.project.Project.release`); it uses the standard library only, so a
model package which defines its project imports it cheaply.

`import sbmlsim` does not import this package.
"""
