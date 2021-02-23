"""Example script for connecting ray."""
import ray


# ray.init(address='10.42.5.3:6379', redis_password='5241590000000000')
ray.init(address="auto", _redis_password="5241590000000000")


@ray.remote
def remote_function():
    """Remote test function."""
    return 1


if __name__ == "__main__":
    for _ in range(4):
        print("executing function")
        remote_function.remote()
