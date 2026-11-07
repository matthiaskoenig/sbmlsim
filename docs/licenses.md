# Licenses

A model is published under two licenses: the model and its documentation under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), the source code which creates and simulates it under [MIT](https://opensource.org/license/MIT). The texts of both licenses travel with the model in its [COMBINE archive](https://combinearchive.org), the `.omex` file which packs the SBML, the documentation and the figures of a model into one file. `sbmlsim.licenses` holds the texts as resources of sbmlsim and writes them into a directory.

The text of CC BY-SA 4.0 is the same for every model. The text of MIT starts with the copyright, which names the years and the authors of the model, e.g. `Copyright (c) 2024-2025 Matthias König`; it is the only part in which the MIT texts of two models differ, so it is the argument `copyright` of the functions, without the words `Copyright (c)`, which the text adds.

## Usage

```python
import tempfile
from pathlib import Path

from sbmlsim.licenses import License, license_text, write_license

results = Path(tempfile.mkdtemp())
cc_by_sa = write_license(results, License.CC_BY_SA_4_0)
mit = write_license(results, License.MIT, copyright="2026 Matthias König")
print(cc_by_sa.name, mit.name)
print(mit.read_text(encoding="utf-8").splitlines()[0])
print(license_text(License.CC_BY_SA_4_0).splitlines()[0])
```

```text title="Output"
cc-by-sa-4.0.txt mit.txt
Copyright (c) 2026 Matthias König
Attribution-ShareAlike 4.0 International
```

The files are named after the license, `cc-by-sa-4.0.txt` and `mit.txt`, and are written in UTF-8 with line feeds as line endings on every platform, so a model creates the same files byte for byte on Linux, macOS and Windows. A license is given as a `License` or as its value (`"mit"`); MIT without a copyright, CC BY-SA 4.0 with one, and a copyright which is empty, spans several lines or starts with `Copyright` are errors.

## The COMBINE archive of a model

The factory of a model writes the texts next to its models and adds them to the entries of the archive. `sbmlsim.release.archive.write_archive(models, out, name=..., copyright=...)` does both: it writes the flat model of a model with submodels, the differential equations of every flat model as markdown and its antimony, the license texts and `<name>.omex` with the README, the licenses, the models and further models and files as entries (the first model, flat, is the master); it returns the files with their entries by the keys of a factory (`README`, `cc-by`, `mit`, the stem of every model), see the [API](api/release.archive.md). A factory which builds its archive itself adds the licenses with [pymetadata](https://github.com/matthiaskoenig/pymetadata):

```py
from pymetadata.omex import EntryFormat, ManifestEntry
from sbmlsim.licenses import License, write_license

entries = {
    license.value: {
        "path": write_license(results_dir, license, copyright=copyright),
        "entry": ManifestEntry(
            location=f"./{license.filename}", format=EntryFormat.TXT, master=False
        ),
    }
    for license, copyright in [
        (License.CC_BY_SA_4_0, None),
        (License.MIT, "2024-2025 Matthias König"),
    ]
}
```

The [release](release.md) of a project writes the archive named by `ReleaseSettings.archive` and attaches it to the GitHub release.
