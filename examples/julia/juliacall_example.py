import numpy as np
from juliacall import Main as jl

A = np.array(jl.rand(5, 3))
x = np.array(jl.randn(3))
y = A @ x

print("juliacall sucessfull")
# print(y)
