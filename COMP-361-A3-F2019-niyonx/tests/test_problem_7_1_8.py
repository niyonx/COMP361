import pytest
from a3 import *
from test_utils import *
from numpy import allclose

def test_problem_7_1_8():
    assert(check_linalg())
    assert(abs(problem_7_1_8(5000)-84.8)<1)
