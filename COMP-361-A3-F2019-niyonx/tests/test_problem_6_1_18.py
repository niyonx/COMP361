import pytest
from a3 import *
from test_utils import *
from numpy import allclose

def test_problem_6_1_18():
    assert(check_linalg())
    assert(abs(problem_6_1_18(1.0)-0.94608)<10E-5)
