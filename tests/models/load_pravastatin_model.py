
from pathlib import Path

import roadrunner
from roadrunner import Config
from sbmlutils import log

Config.setValue(Config.LLVM_BACKEND, Config.LLJIT)
logger = log.get_logger(__name__)

model_path = Path(__file__).parent / "pravastatin_body_flat.xml"
r = roadrunner.RoadRunner(str(model_path))
