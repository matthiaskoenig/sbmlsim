import ray


# ray.init(address='10.42.5.3:6379', redis_password='5241590000000000')
ray.init(address="auto", redis_password="5241590000000000")

# A Ray remote function.
@ray.remote
def remote_function():
    return 1


for _ in range(4):
    print("executing function")
    remote_function.remote()
