import pytest
from a3 import *
from test_utils import *
from numpy import allclose

def test_example_6_12():
    assert(check_linalg())
    assert(abs(example_6_12()-0.85637)<10E-5)
