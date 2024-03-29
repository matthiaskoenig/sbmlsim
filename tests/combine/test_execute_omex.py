"""Testing execution of OMEX."""
from pathlib import Path

import pytest

from sbmlsim.combine.sedml.runner import execute_sedml


# FIXME: fix test data
DATA_DIR = Path(".")


biomodels_omex_paths = []
for path in Path(DATA_DIR / "combine" / "omex" / "biomodels" / "omex").rglob("*.omex"):
    biomodels_omex_paths.append(path)
biomodels_omex_paths = sorted(biomodels_omex_paths)


jws_omex_paths = []
for path in Path(DATA_DIR / "combine" / "omex" / "jws" / "omex").rglob("*.sedx"):
    jws_omex_paths.append(path)
jws_omex_paths = sorted(jws_omex_paths)


l1v3_omex_paths = []
for path in Path(DATA_DIR / "combine" / "omex" / "specification" / "L1V3").rglob(
    "*.omex"
):
    l1v3_omex_paths.append(path)
l1v3_omex_paths = sorted(l1v3_omex_paths)
print(l1v3_omex_paths)


@pytest.mark.skip(reason="no SED-ML support")
@pytest.mark.parametrize("omex_path", biomodels_omex_paths)
def test_biomodel_omex(omex_path: Path, tmp_path: Path) -> None:
    """Biomodel omex."""
    execute_sedml(path=omex_path, working_dir=tmp_path, output_path=tmp_path)


@pytest.mark.skip(reason="no SED-ML support")
@pytest.mark.parametrize("omex_path", jws_omex_paths)
def test_jws_omex(omex_path: Path, tmp_path: Path) -> None:
    """JSW omex."""
    execute_sedml(path=omex_path, working_dir=tmp_path, output_path=tmp_path)


@pytest.mark.skip(reason="no SED-ML support")
@pytest.mark.parametrize("omex_path", l1v3_omex_paths)
def test_l1v3_omex(omex_path: Path, tmp_path: Path) -> None:
    """L1V3 omex."""
    execute_sedml(path=omex_path, working_dir=tmp_path, output_path=tmp_path)


@pytest.mark.skip(reason="no SED-ML support")
def test_combine_archive_showcase_omex(tmp_path: Path) -> None:
    """COMBINE archive showcase omex."""
    omex_path = DATA_DIR / "combine" / "omex" / "CombineArchiveShowCase.omex"
    execute_sedml(path=omex_path, working_dir=tmp_path, output_path=tmp_path)
