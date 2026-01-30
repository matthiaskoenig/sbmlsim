"""
ray start --head --port=6379
"""


import time
tstart = time.time()
import ray

ray.init(address='auto')
# ray.init()


tend = time.time()
print(f"ray start time:", tend-tstart)
