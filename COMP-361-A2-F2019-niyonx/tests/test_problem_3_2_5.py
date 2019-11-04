import pytest
from a2 import *
from test_utils import *

def test_problem_3_2_5():
    assert(check_linalg())
    tol = 10E-4
    assert(abs(problem_3_2_5()-1.8722)<tol)

def test_2020():
    assert(check_linalg())
    tol=10E-3
    assert(abs(extrapolation_3_2_5()-404.8421)<tol)
