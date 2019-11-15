import pytest
from a3 import *
from test_utils import *
from numpy import allclose

def test_problem_8_2_18():
    assert(check_linalg())
    assert(abs(problem_8_2_18(100,57)-37.81)<0.1)
