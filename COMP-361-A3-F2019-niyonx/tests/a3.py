#!/usr/bin/env python
import math

import numpy
from numpy import array, arange, sign

'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''


def python2_vs_python3():
    '''
    A few of you lost all their marks in A2 because their assignment contained
    Python 2 code that does not work in Python 3, in particular print statements
    without parentheses. For instance, 'print hello' is valid in Python 2 but not
    in Python 3.
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer **with Python 3** and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding Python 2 syntax to this file...
    Test: 'tests/test_python3.py'
    '''
    return ("I won't use Python 2 syntax my code",
            "I will always use parentheses around print statements ",
            "I will check the outcome of the tests using pytest or Travis"
            )


'''
    Part 2: Integration (Chapter 6)
'''

from numpy import arange


def problem_6_1_18(x):
    '''
    We will solve problem 6.1.18 in the textbook.
    Task: The function must return the integral of sin(t)/t
          between 0 and x:
              problem_6_1_18(x) = int_0^x{sin(t)/t dt}
    Example: problem_6_1_18(1.0) = 0.94608
    Test: Function 'test_problem_6_1_18' in 'tests/test_problem_6_1_18.py'
    Hints: * use the composite trapezoid rule
           * the integrated function has a singularity in 0. An easy way
             to deal with this is to integrate between a small positive value and x.
    '''

    def f(t):
        return math.sin(t) / t

    def trapezoid(f, a, b, n):
        '''
        Integrates f between a and b using n panels (n+1 points)
        '''
        h = (b - a) / n
        x = a + h * arange(n + 1)
        I = f(x[0]) / 2
        for i in range(1, n):
            I += f(x[i])
        I += f(x[n]) / 2
        return h * I

    return trapezoid(f, 0.000000000001, x, 100)
    raise Exception("Not implemented")


def example_6_12():
    '''
    We will implement example 6.12 in the textbook:
        "
            Evaluate the value of int_1.5^3 f(x)dx ('the integral of f(x)
            between 1.5 and 3'), where f(x) is represented by the
            unevenly spaced data points defined in x_data and y_data.
        "
    Task: This function must return the value of int_1.5^3 f(x)dx where
          f(x) is represented by the evenly spaced data points in x_data and
          y_data below.
    Test: function 'test_example_6_12' in 'tests/test_example_6_12.py'.
    Hints: 1. interpolate the given points by a polynomial of degree 5.
           2. use 3-node Gauss-Legendre integration (with change of variable)
              to integrate the polynomial.
    '''

    x_data = array([1.2, 1.7, 2.0, 2.4, 2.9, 3.3])
    y_data = array([-0.36236, 0.12884, 0.41615, 0.73739, 0.97096, 0.98748])

    # def f(x):
    #     return -1 * math.cos(x)

    def newton_coeffs(x_data, y_data):
        '''
        Returns the coefficients of the Newton polynomial
        '''
        a = y_data.copy()
        m = x_data.size
        assert (m == y_data.size)
        for k in range(1, m):  # go through columns of the table
            for i in range(k, m):  # go through the lines below the diagonal
                a[i] = (a[i] - a[k - 1]) / (x_data[i] - x_data[k - 1])
                # print(i, k, a[i])
            # now a contains column k of the table
        return a

    def swap(a, i, j):
        if len(numpy.shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_substitution(a, b):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        x = numpy.zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - numpy.dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        # New in pivot version
        s = numpy.zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = numpy.argmax(abs(a[k:, k]) / s[k:]) + k
            swap(a, p, k)
            swap(b, p, k)
            swap(s, p, k)
            # The remainder remains as in the previous version
            for i in range(k + 1, n):
                assert (a[k, k] != 0)  # this shouldn't happen now, unless the matrix is singular
                if (a[i, k] != 0):  # no need to do anything when lambda is 0
                    lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                    a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                    b[i] = b[i] - lmbda * b[k]
                if verbose:
                    print(a, b)

    def gauss_pivot(a, b):
        gauss_elimination_pivot(a, b)
        return gauss_substitution(a, b)  # as in the previous version

    def polynomial_fit(x_data, y_data, m):
        '''
        Returns the ai
        '''
        # x_power[i] will contain sum_i x_i^k, k = 0, 2m
        m += 1
        x_powers = numpy.zeros(2 * m)
        b = numpy.zeros(m)
        for i in range(2 * m):
            x_powers[i] = sum(x_data ** i)
            if i < m:
                b[i] = sum(y_data * x_data ** i)
        a = numpy.zeros((m, m))
        for k in range(m):
            for j in range(m):
                a[k, j] = x_powers[j + k]
        return gauss_pivot(a, b)[::-1]

    def f(x):
        coeff = polynomial_fit(x_data, y_data, 5)[::-1]
        return coeff[0] + coeff[1] * x + coeff[2] * (x ** 2) + coeff[3] * (x ** 3) + coeff[4] * (x ** 4) + coeff[5] * (
                    x ** 5)

    def gaussQuad(f):
        return (1.5 / 2) * ((f(1.6691) * 0.555556) + (f(2.25) * 0.888889) + (f(2.8309) * 0.555556))

    return gaussQuad(f)
    raise Exception("Not implemented")


# print(example_6_12())

'''
    Part 3: Initial-Value Problems
'''


def problem_7_1_8(x):
    '''
    We will solve problem 7.1.8 in the textbook. A skydiver of mass m in a
    vertical free fall experiences an aerodynamic drag force F=cy'² ('c times
    y prime square') where y is measured downward from the start of the fall,
    and y is a function of time (y' denotes the derivative of y w.r.t time).
    The differential equation describing the fall is:
         y''=g-(c/m)y'²
    And y(0)=y'(0)=0 as this is a free fall.
    Task: The function must return the time of a fall of x meters, where
          x is the parameter of the function. The values of g, c and m are
          given below.
    Test: function 'test_problem_7_1_8' in 'tests/test_problem_7_1_8.py'
    Hint: use Runge-Kutta 4.
    '''

    # g = 9.80665  # m/s**2
    # c = 0.2028  # kg/m
    # m = 80  # kg

    def runge_kutta_4(F, x0, y0, x, h):
        '''
        Return y(x) given the following initial value problem:
        y' = F(x, y)
        y(x0) = y0 # initial conditions
        h is the increment of x used in integration
        F = [y'[0], y'[1], ..., y'[n-1]]
        y = [y[0], y[1], ..., y[n-1]]
        '''
        X = []
        Y = []
        X.append(x0)
        Y.append(y0)
        while x0 < x:
            k0 = F(x0, y0)
            k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
            k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
            k3 = F(x0 + h, y0 + h * k2)
            y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
            x0 += h
            X.append(x0)
            Y.append(y0)
        return array(X), array(Y)

    g = 9.80665
    c_D = 0.2028
    m = 80

    def F(x, y):
        F = numpy.zeros((2,), dtype=float)
        F[0] = y[1]
        F[1] = - (g - (c_D / m) * y[1] ** 2)
        return F

    x0 = 0.0  # start of the integration
    xStop = 100  # end of the integration
    y0 = numpy.asarray([x, 0.], dtype=float)
    h = 0.1
    freq = 20

    X, Y = runge_kutta_4(F, x0, y0, xStop, h)

    y_sort_inds = numpy.argsort(Y[:, 0])
    y_sorted = Y[y_sort_inds, 0]
    x_sorted = X[y_sort_inds]

    def swap(a, i, j):
        if len(numpy.shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_substitution(a, b):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        x = numpy.zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - numpy.dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        # New in pivot version
        s = numpy.zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = numpy.argmax(abs(a[k:, k]) / s[k:]) + k
            swap(a, p, k)
            swap(b, p, k)
            swap(s, p, k)
            # The remainder remains as in the previous version
            for i in range(k + 1, n):
                assert (a[k, k] != 0)  # this shouldn't happen now, unless the matrix is singular
                if (a[i, k] != 0):  # no need to do anything when lambda is 0
                    lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                    a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                    b[i] = b[i] - lmbda * b[k]
                if verbose:
                    print(a, b)

    def gauss_pivot(a, b):
        gauss_elimination_pivot(a, b)
        return gauss_substitution(a, b)  # as in the previous version

    def polynomial_fit(x_data, y_data, m):
        '''
        Returns the ai
        '''
        # x_power[i] will contain sum_i x_i^k, k = 0, 2m
        m += 1
        x_powers = numpy.zeros(2 * m)
        b = numpy.zeros(m)
        for i in range(2 * m):
            x_powers[i] = sum(x_data ** i)
            if i < m:
                b[i] = sum(y_data * x_data ** i)
        a = numpy.zeros((m, m))
        for k in range(m):
            for j in range(m):
                a[k, j] = x_powers[j + k]
        return gauss_pivot(a, b)[::-1]

    def f(x):
        return numpy.polyval(polynomial_fit(x_sorted, y_sorted, 3), x)

    def false_position(f, a, b, delta_x, max_iter=1000):
        '''
        f is the function for which we will find a zero
        a and b define the bracket
        delta_x is the desired accuracy
        Returns ci such that |ci-c_{i-1}| < delta_x
        '''
        fa = f(a)
        fb = f(b)
        if numpy.sign(fa) == numpy.sign(fb):
            raise Exception("Root hasn't been bracketed")
        estimates = []
        for i in range(max_iter):
            c = (a * fb - b * fa) / (fb - fa)
            estimates.append(c)
            fc = f(c)
            if numpy.sign(fc) == numpy.sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc
            if len(estimates) >= 2 and abs(estimates[-1] - estimates[-2]) <= delta_x:
                break
        return c

    return false_position(f, 0, 100, 0.001, 1000)
    raise Exception("Not implemented")


# print(problem_7_1_8(5000))


def problem_7_1_11(x):
    '''
    We will solve problem 7.1.11 in the textbook.
    Task: this function must return the value of y(x) where y is solution of the
          following initial-value problem:
            y' = sin(xy), y(0) = 2
    Test: function 'test_problem_7_1_11' in 'test/test_problem_7_1_11.py'
    Hint: Use Runge-Kutta 4.
    '''

    def F(x, y):
        return math.sin(x * y)

    def runge_kutta_4(F, x0, y0, x, h):
        '''
        Return y(x) given the following initial value problem:
        y' = F(x, y)
        y(x0) = y0 # initial conditions
        h is the increment of x used in integration
        F = [y'[0], y'[1], ..., y'[n-1]]
        y = [y[0], y[1], ..., y[n-1]]
        '''
        X = []
        Y = []
        X.append(x0)
        Y.append(y0)
        while x0 < x:
            k0 = F(x0, y0)
            k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
            k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
            k3 = F(x0 + h, y0 + h * k2)
            y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
            x0 += h
            X.append(x0)
            Y.append(y0)
        return array(X), array(Y)

    X, Y = runge_kutta_4(F, 0, 2, 6, 0.01)

    def swap(a, i, j):
        if len(numpy.shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_substitution(a, b):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        x = numpy.zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - numpy.dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        # New in pivot version
        s = numpy.zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = numpy.argmax(abs(a[k:, k]) / s[k:]) + k
            swap(a, p, k)
            swap(b, p, k)
            swap(s, p, k)
            # The remainder remains as in the previous version
            for i in range(k + 1, n):
                assert (a[k, k] != 0)  # this shouldn't happen now, unless the matrix is singular
                if (a[i, k] != 0):  # no need to do anything when lambda is 0
                    lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                    a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                    b[i] = b[i] - lmbda * b[k]
                if verbose:
                    print(a, b)

    def gauss_pivot(a, b):
        gauss_elimination_pivot(a, b)
        return gauss_substitution(a, b)  # as in the previous version

    def polynomial_fit(x_data, y_data, m):
        '''
        Returns the ai
        '''
        # x_power[i] will contain sum_i x_i^k, k = 0, 2m
        m += 1
        x_powers = numpy.zeros(2 * m)
        b = numpy.zeros(m)
        for i in range(2 * m):
            x_powers[i] = sum(x_data ** i)
            if i < m:
                b[i] = sum(y_data * x_data ** i)
        a = numpy.zeros((m, m))
        for k in range(m):
            for j in range(m):
                a[k, j] = x_powers[j + k]
        return gauss_pivot(a, b)[::-1]

    def f(x):
        return numpy.polyval(polynomial_fit(X, Y, 3), x)

    return f(x)
    raise Exception("Not implemented")


# print(problem_7_1_11(2))
'''
    Part 4: Two-Point Boundary Value Problems
'''


def problem_8_2_18(a, r0):
    '''
    We will solve problem 8.2.18 in the textbook. A thick cylinder of
    radius 'a' conveys a fluid with a temperature of 0 degrees Celsius in
    an inner cylinder of radius 'a/2'. At the same time, the outer cylinder is
    immersed in a bath that is kept at 200 Celsius. The goal is to determine the
    temperature profile through the thickness of the cylinder, knowing that
    it is governed by the following differential equation:
        d²T/dr²  = -1/r*dT/dr
        with the following boundary conditions:
            T(r=a/2) = 0
            T(r=a) = 200
    Task: The function must return the value of the temperature T at r=r0
          for a cylinder of radius a (a/2<=r0<=a).
    Test:  Function 'test_problem_8_2_18' in 'tests/test_problem_8_2_18'
    Hints: Use the shooting method. In the shooting method, use h=0.01
           in Runge-Kutta 4.
    '''

    def shooting_o2(F, a, alpha, b, beta, u0, u1, delta=10E-3):
        '''
        Solve the boundary condition problem defined by:
        y' = F(x, y)
        y(a) = alpha
        y(b) = beta
        u0 and u1 define a bracket for y'(a)
        delta is the desired accuracy on y'(a)
        Assumes problem is of order 2 (F has two coordinates, alpha and beta are scalars)
        '''

        def r(u):
            '''
            Boundary residual, as in equation (1)
            '''
            # Estimate theta_u
            # Evaluate y and y' until x=b, using initial condition y(a)=alpha and y'(a)=u
            X, Y = runge_kutta_4(F, a, array([alpha, u]), b, 0.2)
            theta_u = Y[-1, 0]  # last row, first column (y)
            return theta_u - beta

        # Find u as a the zero of r
        u, _ = false_position(r, u0, u1, delta)

        # Now use u to solve the initial value problem one more time
        X, Y = runge_kutta_4(F, a, array([alpha, u]), b, 0.2)
        return X, Y

    def false_position(f, a, b, delta_x):
        '''
        f is the function for which we will find a zero
        a and b define the bracket
        delta_x is the desired accuracy
        Returns ci such that |ci-c_{i-1}| < delta_x
        '''
        fa = f(a)
        fb = f(b)
        if sign(fa) == sign(fb):
            raise Exception("Root hasn't been bracketed")
        estimates = []
        while True:
            c = (a * fb - b * fa) / (fb - fa)
            estimates.append(c)
            fc = f(c)
            if sign(fc) == sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc
            if len(estimates) >= 2 and abs(estimates[-1] - estimates[-2]) <= delta_x:
                break
        return c, estimates

    def runge_kutta_4(F, x0, y0, x, h):
        X = []
        Y = []
        X.append(x0)
        Y.append(y0)
        while x0 < x:
            k0 = F(x0, y0)
            k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
            k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
            k3 = F(x0 + h, y0 + h * k2)
            y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
            x0 += h
            X.append(x0)
            Y.append(y0)
        return array(X), array(Y)

    def F(x, y):
        return array(y[1], (-1 / x) * y[1])

    X, Y = shooting_o2(F, a / 2, 0, a, 200, 0, 100)

    def swap(a, i, j):
        if len(numpy.shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_substitution(a, b):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        x = numpy.zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - numpy.dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = numpy.shape(a)
        n2, = numpy.shape(b)
        assert (n == n2)
        # New in pivot version
        s = numpy.zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = numpy.argmax(abs(a[k:, k]) / s[k:]) + k
            swap(a, p, k)
            swap(b, p, k)
            swap(s, p, k)
            # The remainder remains as in the previous version
            for i in range(k + 1, n):
                assert (a[k, k] != 0)  # this shouldn't happen now, unless the matrix is singular
                if (a[i, k] != 0):  # no need to do anything when lambda is 0
                    lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                    a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                    b[i] = b[i] - lmbda * b[k]
                if verbose:
                    print(a, b)

    def gauss_pivot(a, b):
        gauss_elimination_pivot(a, b)
        return gauss_substitution(a, b)  # as in the previous version

    def polynomial_fit(x_data, y_data, m):
        '''
        Returns the ai
        '''
        # x_power[i] will contain sum_i x_i^k, k = 0, 2m
        m += 1
        x_powers = numpy.zeros(2 * m)
        b = numpy.zeros(m)
        for i in range(2 * m):
            x_powers[i] = sum(x_data ** i)
            if i < m:
                b[i] = sum(y_data * x_data ** i)
        a = numpy.zeros((m, m))
        for k in range(m):
            for j in range(m):
                a[k, j] = x_powers[j + k]
        return gauss_pivot(a, b)[::-1]

    def f(x):
        return numpy.polyval(polynomial_fit(X, Y, 3), x)

    return 0
    raise Exception("Not implemented")


print(problem_8_2_18(100, 57))
