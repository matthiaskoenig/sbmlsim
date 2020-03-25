# Additional mathml functions.

def sqr(x):
    return x*x


def root(a, b):
    return a**(1/b)


def xor(*args):
    foundZero = 0
    foundOne = 0
    for a in args:
        if not a:
           foundZero = 1
        else:
           foundOne = 1
    if foundZero and foundOne:
        return 1
    else:
        return 0


def piecewise(*args):
    Nargs = len(args)
    for k in range(0, Nargs-1, 2):
        if args[k+1]:
            return args[k]
    else:
        return args[Nargs-1]

"""
def pow(x, y):
    return x**y

def gt(a, b):
   if a > b:
   	  return 1
   else:
      return 0

def lt(a, b):
   if a < b:
   	  return 1
   else:
      return 0

def geq(a, b):
   if a >= b:
   	  return 1
   else:
      return 0

def leq(a, b):
   if a <= b:
   	  return 1
   else:
      return 0

def neq(a, b):
   if a != b:
   	  return 1
   else:
      return 0

def f_not(a):
   if a == 1:
   	  return 0
   else:
      return 1

def f_and(*args):
    for a in args:
       if a != 1:
          return 0
    return 1

def f_or(*args):
    for a in args:
       if a != 0:
          return  1
    return 0
"""