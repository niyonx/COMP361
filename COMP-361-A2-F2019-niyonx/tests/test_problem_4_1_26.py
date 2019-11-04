import pytest
from a2 import *
from test_utils import *
from numpy import allclose

def test_f():
    assert(check_linalg())
    x_data = array([0.5, 1, 1.5])
    y_data = array([2, 2.5, 2])
    x = array([1, 2, 0.5])
    assert(allclose(f_4_1_26(x_data, y_data, x),
           array([0, 0, 0])))
    

def test_problem():
    assert(check_linalg())
    assert(allclose(problem_4_1_26(array([0.5, 1, 1.5]), array([2, 2.5, 2])),
                    array([1, 2, 0.5])))
