"""Validation of OMEX."""
from pathlib import Path

from biosimulators_utils.combine.io import CombineArchiveReader
from biosimulators_utils.combine.validation import validate
from biosimulators_utils.utils.core import flatten_nested_list_of_strings

from tests import DATA_DIR


repressilator_omex = DATA_DIR / "combine" / "omex" / "tellurium" / "repressilator.omex"
working_dir = Path(__file__).parent / "results" / "repressilator_omex"

archive = CombineArchiveReader().run(
    in_file=str(repressilator_omex), out_dir=str(working_dir)
)
errors, warnings = validate(archive, working_dir)

print(flatten_nested_list_of_strings(errors))
print(flatten_nested_list_of_strings(warnings))


# TODO: add KISAO
