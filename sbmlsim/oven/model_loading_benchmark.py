from sbmlsim.tests.constants import MODEL_GLCWB
import time
import roadrunner

model_path = MODEL_GLCWB

def message(info, time):
    print(f"{info:<20}: {time:4.3f}")

start_time = time.time()
with open(model_path, "r") as f_in:
    sbml_str = f_in.read()
r = roadrunner.RoadRunner(sbml_str)
load_time = time.time() - start_time
message("String loading", load_time)

start_time = time.time()
r = roadrunner.RoadRunner(model_path)
load_time = time.time() - start_time
message("Path loading", load_time)



