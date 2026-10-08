"""Check that the package works as `pip install sbmlsim` installs it.

The environment of the tests has the extras of `dev`, which bring packages the
package itself may need without declaring them; h5py, the backend of the
netCDF files of a result, was one. This script runs a scan and writes its
result as netCDF and reads it back, in an install without extras:

```bash
uv venv plain
uv pip install --python plain/bin/python .
plain/bin/python -I scripts/plain_install.py
```

The job `plain install` of the workflow `ci-cd.yml` runs it.
"""

import tempfile
from pathlib import Path

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator


def main() -> None:
    """Run a scan and write its result as netCDF and read it back."""
    scan = Scan(
        Simulation(end=10, steps=10),
        [Dimension("dim_n", values={"n": [2.0, 3.0]})],
    )
    result = Simulator(n_workers=1).run(REPRESSILATOR_SBML, scan)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.nc"
        result.to_netcdf(path)
        again = ScanResult.from_netcdf(path)
    if not again.ds.identical(result.ds):
        raise SystemExit("The result read from netCDF differs from the one written.")
    print(f"netCDF round trip of {again}")


if __name__ == "__main__":
    main()
