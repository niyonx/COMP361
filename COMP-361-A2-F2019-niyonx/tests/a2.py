#!/usr/bin/env python

from numpy import array, arange, sign, mean, zeros, shape, dot, argmax
import math

'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''


def spaces_and_tabs():
    '''
    A few of you lost all their marks in A1 because their file
    contained syntax errors such as:
      + Missing ':' at the end of 'if' or 'for' statements.
      + A mix of tabs and spaces to indent code. Remember that indendation is part
        of Python's syntax. Mixing tabs and spaces is not good because it
        makes indentation levels ambiguous. For this reason, files
        containing a mix of tabs and spaces are invalid in Python 3. This
        file uses *spaces* for indentation: don't add tabs to it!
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding syntax errors to this file...
    Test: 'tests/test_spaces_and_tabs.py'
    '''
    return ("I won't use tabs in my code",
            "I will make sure that my code has no syntax error",
            "I will check the outcome of the tests using pytest or Travis"
            )


'''
    Part 2: Linear regression
'''


def problem_3_2_5():
    '''
    We will solve problem 3.2.5 in the textbook.
    Arrays 'year' and 'ppm' contain the annual atmospheric CO2 concentration
    in parts per million in Antarctica.
    Task: This function must return the average increase in ppm per year,
          obtained by fitting a straight line to the data.
    Test: Function 'test_problem' in 'tests/test_problem_3_2_5.py'
    Hint: Fitting is meant in the least-square sense.
    '''

    year = arange(1994, 2010)  # from 1994 to 2009
    ppm = array([356.8, 358.2, 360.3, 361.8, 364.0, 365.7, 366.7, 368.2,
                 370.5, 372.2, 374.9, 376.7, 378.7, 381.0, 382.9, 384.7])

    # ppmAvg = sum(ppm) / len(ppm)
    # yearAvg = sum(year) / len(year)
    # bNum = 0
    # bDen = 0
    # for x in range(len(year)):
    #     bNum += year[x] * (ppm[x] - ppmAvg)
    #     bDen += ppm[x] * (ppm[x] - ppmAvg)
    # b = bNum / bDen
    # a = yearAvg - ppmAvg * b
    # # fx = a + bx
    # avgInc = 0
    # for x in range(1, len(year)):
    #     avgInc += ((year[x] - a) / b) - ((year[x - 1] - a) / b)
    # avgInc = avgInc / (len(year) - 1)
    # return avgInc

    xbar = mean(year)
    b = sum(ppm * (year - xbar)) / sum(year * (year - xbar))
    return b
    raise Exception("Not implemented")


# print(problem_3_2_5())


def extrapolation_3_2_5():
    '''
    Task: Return the estimated atmospheric CO2 concentration in Antarctica
          in 2020.
    Test: Function 'test_2020' in 'tests/test_problem_3_2_5.py'
    Hint: Use the result of the previous function.
    '''

    year = arange(1994, 2010)  # from 1994 to 2009
    ppm = array([356.8, 358.2, 360.3, 361.8, 364.0, 365.7, 366.7, 368.2,
                 370.5, 372.2, 374.9, 376.7, 378.7, 381.0, 382.9, 384.7])

    # ppmAvg = sum(ppm) / len(ppm)
    # yearAvg = sum(year) / len(year)
    # bNum = 0
    # bDen = 0
    # for x in range(len(year)):
    #     bNum += year[x] * (ppm[x] - ppmAvg)
    #     bDen += ppm[x] * (ppm[x] - ppmAvg)
    # b = bNum / bDen
    # a = yearAvg - ppmAvg * b
    # # fx = a + bx
    # return ((2020 - a) / b)
    a = mean(ppm) - mean(year) * problem_3_2_5()
    return a + (problem_3_2_5() * 2020)
    raise Exception("Not implemented")


# print(extrapolation_3_2_5())
'''
    Part 3: Non-linear equations
'''

'''
    We will solve problem 4.1.19 in the textbook:
    "
        The speed v of a Saturn V rocket in vertical flight near the surface
        of earth can be approximated by:
            v = u*log(M0/(M0-mdot*t))-g*t
            (log base is e)
        where:
           * u = 2510 m /s is the velocity of exhaust relative to the rocket
           * M0 = 2.8E6 kg is the mass of the rocket at liftoff
           * mdot = 13.3E3 kg/s is the rate of fuel consumption
           * g = 9.81 m/s**2 is the gravitational acceleration
           * t is the time measured from liftoff
    "
'''


def f_and_df(t):
    '''
    Task: return a tuple containing (1) the value of the velocity v
          at time t, (2) the value of the derivative of v at time t.
    Parameter: 't' is the value at which the function and its derivative
               must be evaluated.
    Example: f_and_df(100) must be close to (636.3361111401882, 12.89952381017656)
    Test: function 'f_and_df' in 'tests/test_problem_4_1_19'
    Hint: to compute f', use the central approximation
    '''

    u = 2510
    M0 = 2.8E6
    mdot = 13.3E3
    g = 9.81
    # check for h = E102
    v = u * math.log(M0 / (M0 - mdot * t)) - g * t
    # v'(t) = (f(t+delta_t)-f(t-delta_t))/2*delta_t
    dv = ((u * math.log(M0 / (M0 - mdot * (t + 0.1))) - g * (t + 0.1)) - (
            u * math.log(M0 / (M0 - mdot * (t - 0.1))) - g * (t - 0.1))) / (2 * 0.1)
    return v, dv
    raise Exception("Not implemented")


# print(f_and_df(100))

def newton_raphson(f, diff, init_x, tol, max_iter=1000):
    '''
    f is the function for which a zero is seeked
    diff is the derivative of the function
    init_x is the initial estimate
    tol is the tolerance (accuracy) of the solution
    max_iter is the desired maximal number of iterations
    '''
    x = init_x
    for i in range(max_iter):  # we will break out of the loop when we find the root
        delta_x = -f(x) / diff(x)
        x = x + delta_x
        if abs(delta_x) <= tol:
            return x
    raise Exception("Unable to find a root")


def problem_4_1_19(v1, acc):
    '''
    Task: return the time in seconds when the rocket reaches velocity v1,
          with accuracy acc.
    Parameters:  'v1' is a float representing the velocity of the rocket in m/s.
                 'acc' is a float representing the accuracy of the solution.
    Example: problem_4_1_19(335, 0.1) = 70.877972
    Test: function 'test_problem' in 'tests/test_problem_4_1_9.py'
    Hint: plot the function to get a first guess at the solution.
    '''

    u = 2510
    M0 = 2.8E6
    mdot = 13.3E3
    g = 9.81

    def f(t):
        return (u * math.log(M0 / (M0 - mdot * t)) - g * t) - v1

    def df(t):
        return (u * mdot) / (M0 - mdot * t) - g

    # xs = [x for x in range(10, 100, 10)]
    # fs = [f(t, u, M0, mdot, g,v1) for t in range(10, 100, 10)]
    # t, estimates = root_find_bisection(f(t, u, M0, mdot, g, v1), 2, 4, 0.01)
    # print(fs)
    # v, estimates = newton_raphson(f(t, u, M0, mdot, g, v1), df(t, u, M0, mdot, g), 3, acc)
    # return v

    return newton_raphson(f, df, 0, acc)
    raise Exception("Not implemented")


# print(problem_4_1_19(335, 0.1))

'''
    Part 4: Systems of non-linear equations
'''

'''
    We will solve problem 4.1.26 from the textbook:
        "
        The equation of a circle is: (x-a)**2 + (y-b)**2 = R**2
        where R is the radius and (a,b) are the coordinates of the center.
        Given the coordinates of three points p1, p2 and p3, find a, b
        and R such that the circle of center (a, b) and radius R passes
        by p1, p2 and p3.
        "
'''


def f_4_1_26(x_data, y_data, x):
    '''
    The problem consists in finding the zero of a
    function of 3 variables (a, b and R).
    Task: return an array containing the coordinates of f(x) such
          that the problem can be solved by finding a zero of f.
          x is a vector representing (a, b, R).
    Parameters: + 'x_data' is an array of 3 coordinates representing the abscissa
                   of the input points.
                + 'y_data' is an array of 3 coordinates representing the ordinates
                   of the input points.
                + 'x' is an array of 3 coordinates representing (a, b, R)
    Example: f_4_1_26(array([0.5, 1, 1.5]),
                      array([2, 2.5, 2]),
                      array([1, 2, 0.5])) = [0, 0, 0]
    Test: function 'test_f' in 'tests/test_problem_4_1_26.py'
    '''
    s = [1, 1, 1]
    for i in range(len(x_data)):
        if (x_data[i] - x[0]) ** 2 + (y_data[i] - x[1]) ** 2 == x[2] ** 2:
            s[i] = 0
    return s
    raise Exception("Not implemented")


# print(f_4_1_26(array([0.5, 1, 1.5]), array([2, 2.5, 2]), array([1, 2, 0.5])))

def problem_4_1_26(x_data, y_data):
    '''
    Task: return a, b and R so that the circle of center (a, b) and radius
          R passes by the 3 points defined by x_data and y_data.
    Parameters: + 'x_data' is an array of 3 coordinates representing the abscissa
                   of the input points.
                + 'y_data' is an array of 3 coordinates representing the ordinates
                   of the input points.
    Example: problem_4_1_26(array([0.5, 1, 1.5]), array([2, 2.5, 2]))
             must return [1., 2., 0.5]
    Test: function 'test_problem' in 'tests/test_problem_4_1_26.py''
    Hint: use Newton-Raphson for systems of non-linear equations to find
          a zero of f_4_1_26.
    '''

    def jacobian(f, x):
        '''
        Returns the Jacobian matrix of f taken in x J(x)
        '''
        n = len(x)
        jac = zeros((n, n))
        h = 10E-4
        fx = f(x)
        # go through the columns of J
        for j in range(n):
            # compute x + h ej
            old_xj = x[j]
            x[j] += h
            # update the Jacobian matrix (eq 3)
            # Now x is x + h*ej
            jac[:, j] = (f(x) - fx) / h
            # restore x[j]
            x[j] = old_xj
        return jac

    def gauss_substitution(a, b):
        n, m = shape(a)
        n2, = shape(b)
        assert (n == n2)
        x = zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def swap(a, i, j):
        if len(shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = shape(a)
        n2, = shape(b)
        assert (n == n2)
        # New in pivot version
        s = zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = argmax(abs(a[k:, k]) / s[k:]) + k
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

    def newton_raphson_system(f, init_x, epsilon=10E-10, max_iterations=1000):
        '''
        Return a solution of f(x)=0 by Newton-Raphson method.
        init_x is the initial guess of the solution
        '''
        x = init_x
        for i in range(max_iterations):
            J = jacobian(f, x)
            delta_x = gauss_pivot(J, -f(x))  # we could also use our functions from Chapter 2!
            x = x + delta_x
            if math.sqrt(sum(delta_x ** 2)) <= epsilon:
                print("Converged in {} iterations".format(i))
                return x
        raise Exception("Could not find root!")

    def f(x):
        f = zeros((len(x)))
        f[0] = (x_data[0] - x[0]) ** 2 + (y_data[0] - x[1]) ** 2 - x[2] ** 2
        f[1] = (x_data[1] - x[0]) ** 2 + (y_data[1] - x[1]) ** 2 - x[2] ** 2
        f[2] = (x_data[2] - x[0]) ** 2 + (y_data[2] - x[1]) ** 2 - x[2] ** 2
        return f

    a0 = mean(x_data)
    b0 = mean(y_data)

    r0s = []
    for x, y in zip(x_data, y_data):
        r0s.append(math.sqrt((x - a0) ** 2 + (y - b0) ** 2))
    R0 = mean(r0s)

    return newton_raphson_system(f, [a0, b0, R0])
    raise Exception("Not implemented")


'''
    Part 5: Interpolation and Numerical differentiation
'''

'''
    We will solve problem 5.1.11 from the textbook:
        " 1. Use polynomial interpolation to compute f' and f'' at x using
          the data in x_data and y_data:
                x_data = array([-2.2, -0.3, 0.8, 1.9])
                y_data = array([15.180, 10.962, 1.920, -2.040])
          2. Given that f(x) = x**3 - 0.3*x**2 -
             8.56*x + 8.448, gauge the accuracy of the result."
'''


def interpolant_5_1_11():
    '''
    Task: return an array containing the coefficients of the polynomial
          of degree 3 interpolating the data points.
    Test: function 'test_interpolant' of 'tests/test_problem_5_1_11.py'
    Hint: use code from Chapters 2 and 3.
    '''

    def swap(a, i, j):
        if len(shape(a)) == 1:
            a[i], a[j] = a[j], a[i]  # unpacking
        else:
            a[[i, j], :] = a[[j, i], :]

    def gauss_substitution(a, b):
        n, m = shape(a)
        n2, = shape(b)
        assert (n == n2)
        x = zeros(n)
        for i in range(n - 1, -1, -1):  # decreasing index
            x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        return x

    def gauss_elimination_pivot(a, b, verbose=False):
        n, m = shape(a)
        n2, = shape(b)
        assert (n == n2)
        # New in pivot version
        s = zeros(n)
        for i in range(n):
            s[i] = max(abs(a[i, :]))
        for k in range(n - 1):
            # New in pivot version
            p = argmax(abs(a[k:, k]) / s[k:]) + k
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
        x_powers = zeros(2 * m)
        b = zeros(m)
        for i in range(2 * m):
            x_powers[i] = sum(x_data ** i)
            if i < m:
                b[i] = sum(y_data * x_data ** i)
        a = zeros((m, m))
        for k in range(m):
            for j in range(m):
                a[k, j] = x_powers[j + k]
        return gauss_pivot(a, b)

    x_data = array([-2.2, -0.3, 0.8, 1.9])
    y_data = array([15.180, 10.962, 1.920, -2.040])

    return polynomial_fit(x_data, y_data, 3)
    raise Exception("Not implemented")


# print(interpolant_5_1_11())

def d_dd_5_1_11(x):
    '''
    Task: return a tuple containing the value of f' and f'' at x.
    Parameter: x is the value at which f' and f'' must be computed.
    Test: function 'test_d_dd' of 'tests/test_problem_5_1_11.py'
    Example: d_dd_5_1_11(0) must return (8.56, -0.6).
    Hint: differentiate the interpolant returned by the previous function.
    '''

    n = 3
    coeffs = interpolant_5_1_11()
    d_coeffs = zeros(n)
    for i in range(n):
        d_coeffs[i] = (i + 1) * coeffs[i + 1]  # differentiation from coefficients

    n = 2
    coeffs = d_coeffs
    dd_coeffs = zeros(n)
    for i in range(n):
        dd_coeffs[i] = (i + 1) * coeffs[i + 1]  # differentiation from coefficients

    return (d_coeffs[0] + d_coeffs[1] * x + d_coeffs[2] * (x ** 2)), (dd_coeffs[0] + dd_coeffs[1] * x)
    raise Exception("Not implemented")


# print(d_dd_5_1_11(0))


def error_5_1_11(x):
    '''
    Task: return a tuple containing the errors made by your previous
           approximation of f' and f'' at x.
    Parameter:  x is the value at which f' and f'' must be computed.
    Test: function 'test_error' of  'tests/test_problem_5_1_11.py'
    Example: error(0) must return
    Hint: differentiate x**3 - 0.3*x**2 - 8.56*x + 8.448 manually and compare
          the result to the output of the previous function.
    '''

    def df(x): return 3 * (x ** 2) - (0.6 * x) - 8.56

    def ddf(x): return 6 * x - 0.6

    a, b = d_dd_5_1_11(x)
    print(df(x), ddf(x))
    return a - df(x), b - ddf(x)
    raise Exception("Not implemented")

# print(error_5_1_11(0))
