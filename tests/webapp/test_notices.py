"""Tests of the notices of the components of a web app as plain text."""

import pytest

from sbmlsim.webapp.notices import plain_notices


def test_a_heading_is_underlined() -> None:
    text = plain_notices("# Third-party notices\n\n## License texts\n\n### SUNDIALS\n")
    assert text.splitlines() == [
        "Third-party notices",
        "===================",
        "",
        "License texts",
        "-------------",
        "",
        "SUNDIALS",
    ]


def test_a_table_becomes_a_block_per_row() -> None:
    text = plain_notices(
        "| Library | Version | License |\n"
        "|---|---|---|\n"
        "| KLU (SuiteSparse) | 2.3.6 | LGPL-2.1-or-later |\n"
        "| BTF | 2.3.3 | the same |\n"
    )
    # a block per row, its first cell its heading and every other cell a field of the header, so
    # the table reads at the width of a panel instead of wrapping its columns into each other
    assert text.splitlines() == [
        "KLU (SuiteSparse)",
        "  Version: 2.3.6",
        "  License: LGPL-2.1-or-later",
        "",
        "BTF",
        "  Version: 2.3.3",
        "  License: the same",
    ]


def test_the_text_of_a_license_is_kept_verbatim() -> None:
    text = plain_notices(
        "### SUNDIALS: `third_party/sundials/LICENSE`\n"
        "\n"
        "```text\n"
        "BSD 3-Clause License\n"
        "\n"
        "  * `Redistributions` of source code must retain | the notice\n"
        "```\n"
    )
    assert "```" not in text
    # every line of a license text is the line of its file, the markup of markdown included
    assert "  * `Redistributions` of source code must retain | the notice" in text
    # while the heading around it is rendered
    assert "SUNDIALS: third_party/sundials/LICENSE" in text


def test_inline_code_and_a_link_are_plain() -> None:
    text = plain_notices(
        "The code of `sbmlsim-core` is MIT licensed, see [the routines](https://example.org/r).\n"
    )
    assert (
        text
        == "The code of sbmlsim-core is MIT licensed, see the routines (https://example.org/r)."
    )


def test_the_notices_of_the_engine_carry_no_markup() -> None:
    core = pytest.importorskip("sbmlsim_core")
    text = plain_notices(core.THIRD_PARTY)
    lines = text.splitlines()
    assert not [
        one for one in lines if one.startswith(("# ", "## ", "### ", "|", "|---"))
    ]
    assert "```" not in text
    # the libraries which are linked and their licenses are still in it, as is the LGPL notice
    assert "SUNDIALS" in text
    assert "LGPL-2.1-or-later" in text
    assert "BSD 3-Clause License" in text
    # the heading of the document and of its license texts, underlined
    assert text.startswith("Third-party notices\n===================")
