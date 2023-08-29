from numpy.testing import assert_allclose
from math import isclose
import numpy as np


from logistic import f
from logistic import iterate_f
from logistic_fit import fit_r
import pytest

SEED=np.random.seed(42)

# Add here your test for the logistic map



def test_f_corner_cases():
    # Test cases are (x, r, expected)
    cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)

def test_f_generic_cases():
    # Test cases are (x, r, expected)
    cases = [
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5)
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)

@pytest.mark.parametrize('x,r,expected',[
        (0.1, 2.2, 0.198),
        (0.2, 3.4, 0.544),
        (0.5, 2, 0.5)
    ])

def test_f_generic_cases_param(x,r,expected):
    # Test cases are (x, r, expected)

    result = f(x,r)
    assert_allclose(result, expected)


@pytest.mark.parametrize('x,r,expected',[
        (0, 1.1, 0),
        (1, 3.7, 0),
    ])
def test_f_generic_corner_param(x,r,expected):
    # Test cases are (x, r, expected)
    result = f(x,r)
    assert_allclose(result, expected)

@pytest.mark.parametrize('x,r,it,expected',[
        (0.1,2.2, 1, [0.1,0.198]),
        (0.2,3.4, 4, [0.2,0.544,0.843418,0.449019,0.841163]),
        (0.5,2,3,[0.5,0.5,0.5,0.5])
    ])

def test_iterative_f(x,r,it,expected):
    result=iterate_f(it,x,r)
    assert_allclose(result,expected,rtol=5e-07)

def test_fit_r():
    expected=2
    xs=iterate_f(it=23,x=0.3, r=expected)
    result=fit_r(xs)
    assert isclose(result, expected)==True

@pytest.mark.parametrize('x,it,r,expected',[
        (0.3,23,5.3,5.3),
        (0.5,12,2,2),
        (0.9,50,3.4,3.4)
    ])
def test_fit_r_param(x,it,r,expected):
    
    xs=iterate_f(it,x,r)
    result=fit_r(xs)
    assert isclose(result, expected)==True

def test_fit_r_random():
    x0=np.random.uniform(0,1,100)

    for x in x0:
        xs=iterate_f(100,x,1.5)
        assert  isclose(xs[-1],1/3) 

@pytest.mark.parametrize('x0',[(x)for x in np.random.uniform(0,2,100)] )


def test_fit_r_random_param(x0):

    
    xs=iterate_f(100,x0,1.5)
    assert  isclose(xs[-1],1/3) and isclose(xs[-2],1/3) and isclose(xs[-3],1/3) 