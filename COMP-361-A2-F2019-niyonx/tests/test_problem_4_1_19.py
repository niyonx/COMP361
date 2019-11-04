import pytest
from a2 import *
from test_utils import *
from numpy import allclose

def test_f_and_df():
    assert(check_linalg())
    assert(allclose(
            array(f_and_df(100)), array((636.3361111401882, 12.89952381017656))
            ))

def test_problem_4_1_19():
    assert(check_linalg())
    tol = 10E-2
    assert(abs(problem_4_1_19(335, tol)- 70.8779722)<tol)
