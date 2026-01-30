"""
ray start --head --port=6379
"""

import ray
import time

tstart = time.time()


ray.init(address="auto")
# ray.init()


tend = time.time()
print("ray start time:", tend - tstart)
