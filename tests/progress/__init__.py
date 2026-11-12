"""The tests of `sbmlsim.progress`, a package so that its conftest has a name of its own.

Without it the module of the conftest is `conftest`, the name the conftest of
`tests/fit` has as well, and which of them `sys.modules` holds decides whether
a function of the other can be pickled by its qualified name.
"""
