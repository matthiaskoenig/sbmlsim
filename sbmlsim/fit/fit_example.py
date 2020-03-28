import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def f(x, a, b, c):
    """ Model function

    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return a*x**3 + b*x + c


def residual(p, x, y):
    """ Residual calculation

    :param p:
    :param x:
    :param y:
    :return:
    """
    # TODO: the weighting should be in here.

    return y - f(x, *p)


if __name__ == "__main__":
    # create trainings data
    x = np.linspace(0, 20, 20)
    y = f(x, a=1.5, b=5, c=0.2) + 20*np.random.normal(size=len(x))

    plt.plot(x, y, 'or')

    for k in range(20):

        p0 = [1., 1., 1.] + 2.0 * np.random.normal(size=3)
        p0 = [1.6, 6., 0.1] + 2.0 * np.random.normal(size=3)

        # popt, pcov = optimize.leastsq(residual, p0, args=(x, y))
        popt, pcov = optimize.least_squares(fun=residual, x0=p0, bounds=(-np.inf, np.inf),
                                            args=(x, y))

        print(p0, "->", popt)

        # optimal solution
        xn = np.linspace(0, 20, 200)
        yn = f(xn, *popt)
        plt.plot(xn, yn, color="black", alpha=0.8)
    plt.show()
