from numpy import *
from copy import deepcopy

'''
   PART 1: Warm-up
'''


def example_func():
    '''
      Important: READ THIS CAREFULLY.
      Task: This function is an example, you don't have to modify it.
      Example: Nothing to report here, really.
      Test: This function is is tested in tests/test_example.py
            This test just gives you a bonus, yay!
      Hint: The functions below have to be implemented in Python, without
            using any function from numpy's linear algebra module. In each function, a
            docstring formatted as the present one explains what the
            function must do (Task), gives an example of output
            (Example), explains how it will be evaluated (Test), and
            may give you some hints (Hint).
    '''
    return 'It works!'


def square(a):
    '''
      Task: This function tests if a matrix is square. It returns True
            if a represents a square matrix.
      Parameters: a is a numpy array.
      Example: square(array([[1, 2], [3, 4]])) must return True.
      Test: This function is is tested in tests/test_square.py
      Hint: Use numpy's shape function.
    '''
    n, m = shape(a)
    if n == m:
        return True
    else:
        return False
    raise Exception("Function not implemented")


'''
  Part 2: Resolution of linear systems for polynomial interpolation
'''


def collinear(points):
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if (a == 0):
        return True
    else:
        return False

def fit_poly_2(points):
    '''
      Task: This function finds a polynomial P of degree 2 that passes
            through the 3 points contained in list 'points'. It returns a numpy
            array containing the coefficient of a polynomial P: array([a0, a1, a2]),
            where P(x) = a0 + a1*x + a2*x**2. Every (x, y) in 'points' must
            verify y = a0 + a1*x + a2*x**2.
      Parameters: points is a Python list of 3 pairs representing 2D points.
      Example: fit_poly_2([(0, -1), (1, -2), (2, -9)]) must return array([-1, 2, 3])
      Test: This function is is tested by the following functions in tests/test_fit_poly.py:
            - test_fit_poly_2 tests a basic fit
            - test_fit_poly_raises tests that the function raises an
              AssertionError when the polynomial cannot be fit (for
              instance, 3 points are aligned).
      Hint: This should be done by solving a linear system.
    '''

    assert(not collinear(points))
    a = []
    b = []
    for row in points:
        x, y = row
        b.append(y)
        a.append([1, x, x * x])
    a = array(a)
    b = array(b)
    gauss_elimination(a, b)
    return gauss_substitution(a, b)
    raise Exception("Function not implemented")


def gauss_elimination(a, b, verbose=False):
    n, m = shape(a)
    n2, = shape(b)
    assert (n == n2)
    for k in range(n - 1):
        for i in range(k + 1, n):
            assert (a[k, k] != 0)  # woops, what happens in this case? we'll talk about it later!
            if (a[i, k] != 0):  # no need to do anything when lambda is 0
                lmbda = a[i, k] / a[k, k]  # lambda is a reserved keyword in Python
                a[i, k:n] = a[i, k:n] - lmbda * a[k, k:n]  # list slice operations
                b[i] = b[i] - lmbda * b[k]  # don't forget this step!
            if verbose:
                print(a, b)


def gauss_substitution(a, b):
    n, m = shape(a)
    n2, = shape(b)
    assert (n == n2)
    x = zeros(n)
    for i in range(n - 1, -1, -1):  # decreasing index
        x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
    # print(x)
    return x


def fit_poly(points):
    '''
      Task: This function is a generalization of the previous one. It
            finds a polynomial P of degree n that passes
            through the n+1 points contained in list 'points'. It
            returns a numpy array containing the coefficient of a
            polynomial P: array([a0, a1, ..., an]), where P(x) = a0 +
            a1*x + a2*x**2 + ... + an*x**n. Every (x, y) in 'points'
            must verify y = P(x).
      Parameters: points is a Python list of pairs representing 2D points.
      Examples: fit_poly([(0, -1), (1, -2), (2, -9)]) must return
                array([-1, 2, -3]) (as in the previous function) fit_poly([(0, 2),
                (1, 6), (2, 24), (3, 62)]) must return array([2, -1, 4, 1])
      Test: This function is is tested by the following functions in tests/test_fit_poly.py:
            - test_fit_poly tests a basic fit
            - test_fit_poly_n tests the fit on a random polynomial of degre <= 6.
      Hint: This should be done by solving a linear system.
    '''
    a = []
    b = []
    for row in points:
        aRow = []
        x, y = row
        b.append(y)
        for col in range(len(points)):
            aRow.append(x ** col)
        a.append(aRow)
    a = array(a)
    b = array(b)
    gauss_elimination(a, b)
    return gauss_substitution(a, b)
    raise Exception("Function not implemented")


'''
  Part 3: Tridiagonal systems
'''


# print("Teset tridiag solver:\n"+tridiag_solver_n(5))

def tridiag_solver_n(n):
    '''
      Task: This function returns the solution of the following tridiagonal equations:
            4x[1] - x[2] = 9
            -x[i-1] + 4x[i] - x[i+1] = 5, i=2,....n-1
            -x[n-1] + 4x[n] = 5
            The system of equations is the same
            as in problem 2.2.9 in the Textbook, except that here n is a
            parameter of the function. All correct answers will be accepted,
            but you are strongly encouraged to exploit the tridiagonal nature
            of the system.
      Parameters: n is an integer representing the dimension of the system.
      Examples: tridiag_solver_n(2) must return array([41/15, 29/15])
      Test: This function is is tested by the function in tests/test_tridiag_solver.py.
    '''
    # [[4,-1,0,0,0] = 9
    #  [-1,4,-1,0,0] = 5
    #  [0,-1,4,-1,0] = 5
    #  [0,0,-1,4,-1] = 5
    #  [0,0,0,-1,4]]= 5
    # build matrix, forward sub, backward sub sub

    b = []
    b.append(9)
    for i in range(n - 1):
        b.append(5)

    c = []
    e = []
    for i in range(n - 1):
        c.append(-1)
        e.append(-1)
    c.append(0)
    e.append(0)

    d = []
    for i in range(n):
        d.append(4)

    tridiag_decomp(c, d, e)
    tridiag_solve(c, d, e, b)
    return b
    raise Exception("Function not implemented")


def tridiag_solve(c, d, e, b):  # watch out, input has to be in LU form!
    assert (len(c) == len(d) == len(e) == len(b))
    n = len(c)
    # forward substitution
    for i in range(1, n):
        b[i] = b[i] - c[i - 1] * b[i - 1]  # Here we use b to store y
    # back substitution
    b[n - 1] = b[n - 1] / d[n - 1]  # Here we use b to store x
    for i in range(n - 2, -1, -1):
        b[i] = (b[i] - e[i] * b[i + 1]) / d[i]


def tridiag_decomp(c, d, e):
    assert (len(c) == len(d) == len(e))
    n = len(c)
    for k in range(1, n):
        lambd = c[k - 1] / d[k - 1]
        d[k] -= lambd * e[k - 1]
        c[k - 1] = lambd


'''
  Part 4: Gauss Elimination for more than 1 equation
'''


# na = array([[6, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
# nb = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
# gm = gauss_multiple(na, nb)
# print("Test gauss multiple:\n", gm)

def gauss_multiple(a, b):
    '''
      Task: This function returns the solution of the system written as
            AX=B, where A is an n x n square matrix, and X and B are n x m matrices.
            It is equivalent to solving m systems of the form Ax=b, where
            x and b are column vectors. You have to extend the implementation
            of Gauss elimination presented in the course to work with m constant
            vectors instead of only 1. This is problem 2.1.14 in the textbook.
            It is up to you to decide if your function will modify a and b (the
            tests should work in both cases).
      Parameters: a is a numpy array representing a square matrix. b is a numpy
            array representing a matrix with as many lines as in a.
      Test: This function is is tested by the function test_gauss_multiple in tests/test_gauss_multiple.py.
      Hint: Start from the implementation shown in the course!
    '''

    # [[ 3.  2. -7.] [a1,b1]   [4,3]
    #  [ 2. -1. -4.] [a2,b2] = [2,1]
    #  [-1. -3.  1.]][a3, b3]  [5,3]

    result = zeros(shape(b))
    n, m = shape(b)  # 3 , 2
    for i in range(m):  # 0 1
        resultCol = []
        aTemp = deepcopy(a)
        bCol = []
        for j in range(n):
            bCol.append(b[j][i])
        aTemp = array(aTemp)
        bCol = array(bCol)
        gauss_elimination(aTemp, bCol)
        resultCol = gauss_substitution(aTemp, bCol)
        for k in range(n):
            result[k][i] = resultCol[k]
    return array(result)
    raise Exception("Function not implemented")


# na = array([[0, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
# nb = array([[-14, 22], [36, -18], [6, 7]], dtype=float_)
# from numpy.linalg import solve
# nsolution = solve(na, nb)
# print(nsolution)

def gauss_multiple_pivot(a, b):
    '''
      Task: This function returns the same result as the previous one,
            except that it uses scaled row pivoting.
      Parameters: a is a numpy array representing a square matrix. b is a numpy
            array representing a matrix with as many lines as in a.
      Test: This function is is tested by the function
            test_gauss_multiple_pivot in tests/test_gauss_multiple.py.
    '''

    result = zeros(shape(b))
    n, m = shape(b)  # 3 , 2
    for i in range(m):  # 0 1
        resultCol = []
        aTemp = deepcopy(a)
        bCol = []
        for j in range(n):
            bCol.append(b[j][i])
        aTemp = array(aTemp)
        bCol = array(bCol)
        gauss_elimination_pivot(aTemp, bCol)
        resultCol = gauss_substitution(aTemp, bCol)
        for k in range(n):
            result[k][i] = resultCol[k]
    return array(result)
    raise Exception("Function not implemented")


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


# a = array([[0, -1, 1], [-1, 2, -1], [2, -1, 0]], dtype=float_)
# b = array([0, 0, 1], dtype=float_)
# x = gauss_pivot(a, b)
# print(x)


# print("Test for gauss multiple pivot:\n",gauss_multiple_pivot(na,nb))


def matrix_invert(a):
    '''
      Task: This function returns the inverse of the square matrix a passed
            as a paramter.
      Parameters: a is a numpy array representing a non-singular square matrix.
      Test: This function is is tested by function test_inverse in tests/test_inverse.py
      Hint: Remember that the inverse of A is the solution of n linear systems of n
            equations.
    '''

    # build identity matrix
    IM = zeros(shape(a))
    for i in range(len(a)):
        IM[i][i] = 1
    # solve for AX = I
    return gauss_multiple_pivot(a, array(IM))
    raise Exception("Function not implemented")

# def norm(a): # infinity norm
#     n = 0
#     for row in a:
#         s = sum(abs(row))
#         if s > n:
#             n = s
#     return n
#
# def condition(a):
#     from numpy.linalg import inv # this is cheating :), we'll see later how to compute the inverse of a matrix
#     return norm(a)*norm(inv(a))
#
# ccc = array([[1, -1.001], [2.001, -2]])
# print(condition(ccc))



#
# from numpy.linalg import inv
# a = array([[6, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
# solution = inv(a)
# print(solution)

# a = array([[6, 4, 1], [-4, 6, -4], [1, -4, 6]], dtype=float_)
# inva = matrix_invert(a)
# print("Test for inverse:\n", inva)

# points = [(0, -1), (1, -2), (0, -1)]
# print(collinear(points))

