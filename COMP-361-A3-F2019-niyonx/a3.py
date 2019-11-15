#!/usr/bin/env python

from numpy import array, arange

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

    ## YOUR CODE HERE
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
    
    ## YOUR CODE HERE
    raise Exception("Not implemented")


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
    g = 9.80665  # m/s**2
    c = 0.2028  # kg/m
    m = 80  # kg
  
    ## YOUR CODE HERE
    raise Exception("Not implemented")

def problem_7_1_11(x):
    '''
    We will solve problem 7.1.11 in the textbook.
    Task: this function must return the value of y(x) where y is solution of the
          following initial-value problem:
            y' = sin(xy), y(0) = 2
    Test: function 'test_problem_7_1_11' in 'test/test_problem_7_1_11.py'
    Hint: Use Runge-Kutta 4.
    '''

   ## YOUR CODE HERE
    raise Exception("Not implemented")

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

    ## YOUR CODE HERE
    raise Exception("Not implemented")


