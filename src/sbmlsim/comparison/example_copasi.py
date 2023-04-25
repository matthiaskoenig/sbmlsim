from pathlib import Path

base_path: Path = Path(__file__).parent
model_path = base_path / "resources" / "icg_sd.xml"
print(model_path)

from basico import (
    load_model,
    set_parameters,
    get_parameters
)
load_model(location=str(model_path))

set_parameters("body weight [kg]", initial_value=83.5)
print(get_parameters())
