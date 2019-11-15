import pytest
from a3 import *
from test_utils import *
from numpy import allclose

def test_problem_7_1_11():
    assert(check_linalg())
    assert(abs(problem_7_1_11(2)-2.15)<0.1)
