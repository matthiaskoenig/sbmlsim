import logging
from pathlib import Path

from sbmlsim.model import RoadrunnerSBMLModel
from tests import MODEL_REPRESSILATOR
from sbmlutils.log import get_logger
from sbmlutils.log import set_level_for_all_loggers
from filelock import FileLock
import roadrunner

logger = logging.getLogger(__file__)

set_level_for_all_loggers(level=logging.DEBUG)
# RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)
#
# print("finished")

directory = Path(__file__).parent

model_path = directory / "repressilator.xml"
state_path = directory / Path("repressilator.state")

# delete existing state
if state_path.exists():
    state_path.unlink()

print("-", 80)

logger.debug(f"Load model from SBML: '{model_path.resolve()}'")
r = roadrunner.RoadRunner(str(model_path))
# save state path
if state_path is not None:
    with FileLock(state_path):
        r.saveState(str(state_path))
    logger.debug(f"Save state: '{state_path}'")

logger.debug(f"Load model from state: '{state_path}'")
r = roadrunner.RoadRunner()
with FileLock(state_path):
    r.loadState(str(state_path))
logger.debug(f"Model loaded from state: '{state_path}'")
