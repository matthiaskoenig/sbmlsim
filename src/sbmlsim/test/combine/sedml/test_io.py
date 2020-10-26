from sbmlsim.combine.sedml.io import read_sedml
from sbmlsim.test import DATA_DIR


def test_read_repressilator(tmp_path):
    repressilator_l1v4_sedml = (
        DATA_DIR / "sedml" / "l1v4" / "repressilator" / "repressilator_sedml.xml"
    )
    res = read_sedml(repressilator_l1v4_sedml, tmp_path)
    assert res
