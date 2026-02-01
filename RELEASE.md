# Release information

## Steps to make a new release
* update release notes in `release-notes` with commit
* make sure all tests run (`tox -p`)
* check formating and linting (`ruff check`)
* build documentation (`cd _docs | quartodoc build | quatro render`)
* test bump version (`uvx bump-my-version bump [major|minor|patch] --dry-run -vv`)
* bump version (`uvx bump-my-version bump [major|minor|patch]`)
* `git push --tags` (triggers release)
* `git push`

## Test release

```bash
uv venv --python 3.14
uv pip install sbmlsim
```
