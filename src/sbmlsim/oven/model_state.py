import hashlib
from pathlib import Path
from typing import Optional

import roadrunner
print(f"roadrunner version: {roadrunner.__version__}")

def md5_for_path(path):
    """Calculate MD5 of file content."""

    # Open,close, read file and calculate MD5 on its contents
    with open(path, "rb") as f_check:
        # read contents of the file
        data = f_check.read()
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()

def get_state_path(sbml_path: Path) -> Optional[Path]:
    """Get path of the state file.

    The state file is a binary file which allows fast model loading.
    """
    md5 = md5_for_path(sbml_path)
    return Path(f"{sbml_path}_rr{roadrunner.__version__}_{md5}.state")


sbml_path = "omeprazole_body_flat.xml"
state_path = get_state_path(sbml_path)

# state saving and loading
r = roadrunner.RoadRunner()

if state_path.exists():
    r.loadState(str(state_path))
    print(f"Model loaded from state: '{state_path}'")
else:
    print(f"Load model from SBML: '{sbml_path}'")
    r = roadrunner.RoadRunner(str(sbml_path))
    # save state
    r.saveState(str(state_path))

print(f"Load from state: '{state_path}'")
r.loadState(str(state_path))
print(f"Model loaded from state: '{state_path}'")
