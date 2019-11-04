import pytest
from a2 import *
from test_utils import *
from numpy import allclose

def test_interpolant():
    assert(check_linalg())
    assert(allclose(interpolant_5_1_11(), array([8.448, -8.56, -0.3, 1])))

def test_d_dd():
    assert(check_linalg())
    assert(allclose(array(d_dd_5_1_11(0)), array([-8.56, -0.6])))

def test_error():
    assert(check_linalg())
    (a, b) = error_5_1_11(0)
    assert(abs(a) < 10E-13 and abs(b) < 10E-14)
