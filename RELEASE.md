# Release information


## make release
* make sure all tests run (`tox -p`)
* update release notes in `release-notes`
* bump version (`bumpversion patch` or `bumpversion minor` or `bumpversion major`)
* `git push --tags` (triggers release)
* add release-notes for next version

* test installation in virtualenv from pypi
```
mkvirtualenv test --python=python3.8
(test) pip install sbmlsim
```
