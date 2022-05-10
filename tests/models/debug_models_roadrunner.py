from pathlib import Path

# from filelock import FileLock
import roadrunner

directory = Path(__file__).parent

model_path = directory / "repressilator.xml"
state_path = directory / "repressilator.state"

# delete existing state
if state_path.exists():
    state_path.unlink()


print(f"Load model from SBML: '{model_path.resolve()}'")
r = roadrunner.RoadRunner(str(model_path))
# save state path
if state_path is not None:
    #with FileLock(state_path):
    r.saveState(str(state_path))
    print(f"Save state: '{state_path}'")

print(f"Load model from state: '{state_path}'")
r = roadrunner.RoadRunner()
# with FileLock(state_path):
    r.loadState(str(state_path))
print(f"Model loaded from state: '{state_path}'")
