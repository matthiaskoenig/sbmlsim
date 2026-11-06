"""The SId of a text, a key of an experiment."""

import pytest

from sbmlsim.experiment import to_sid


@pytest.mark.parametrize(
    ("text", "sid"),
    [
        ("task_po", "task_po"),
        ("po 2.5 mg", "po_2p5_mg"),
        ("dset_torasemide-M5_po_2.5", "dset_torasemide_M5_po_2p5"),
        ("tor_iv_chronic renal disease", "tor_iv_chronic_renal_disease"),
        ("Fig1_CIM400, ENA10_B1", "Fig1_CIM400__ENA10_B1"),
        ("µg/ml", "_g_ml"),
        ("1.5.2", "1p5p2"),
        ("a.b", "a_b"),
        ("2.", "2_"),
        (".5", "_5"),
        ("a b", "a_b"),
        ("a‐b", "a_b"),
        ("(a+b)*c&d", "_a_b__c_d"),
        ("", ""),
    ],
)
def test_a_text_becomes_an_sid(text: str, sid: str) -> None:
    assert to_sid(text) == sid
    assert len(to_sid(text)) == len(text)


def test_a_part_of_a_key_becomes_the_same_part_of_its_sid() -> None:
    assert f"task_po_{to_sid('2.5')}" == to_sid("task_po_2.5")
