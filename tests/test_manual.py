#!/usr/bin/env python3

from numpy import isclose, log, tensordot, exp, reshape, eye
from numpy.random import normal
from numpy.linalg import norm
from nxarray import NXArray
import nxarray.linalg as linalg


def test_2_norm() -> None:
    arr1 = normal(size=(2, 3, 4))
    arr2 = normal(size=(3, 3, 4))
    nxarr1 = NXArray(arr1, "i1", "i2", "i3")
    nxarr2 = NXArray(arr2, "i2", "i4", "i3")
    _arr1 = nxarr1.release_normalized_array("i2", "i3", "i1")
    assert isclose(norm(_arr1), 1.0)
    assert isclose(nxarr1.log_norm, log(norm(arr1)))
    arr1_dot_arr2 = tensordot(arr1, arr2, ((1, 2), (0, 2)))
    log_norm_arr1_dot_arr2 = log(norm(arr1_dot_arr2))
    nxarr1_dot_nxarr2 = nxarr1 * nxarr2
    assert isclose(log_norm_arr1_dot_arr2, nxarr1_dot_nxarr2.log_norm)
    assert isclose(
        arr1_dot_arr2 / exp(nxarr1_dot_nxarr2.log_norm),
        nxarr1_dot_nxarr2.release_normalized_array("i1", "i4"),
    ).all()
    print("Test 2-norm: OK")


def test_scalar_mul() -> None:
    arr = normal(size=(2, 3, 4))
    nxarr = NXArray(arr, "i1", "i2", "i3")
    log_norm_nxarr = nxarr.log_norm
    assert not ((nxarr * 2.0) is nxarr)
    assert not ((2.0 * nxarr) is nxarr)
    assert isclose(
        (2.0 * nxarr).release_array("i1", "i3", "i2"), 2.0 * arr.transpose((0, 2, 1))
    ).all()
    assert isclose(
        (nxarr * 2.0).release_array("i1", "i3", "i2"), 2.0 * arr.transpose((0, 2, 1))
    ).all()
    assert isclose((2.0 * nxarr).log_norm - log_norm_nxarr, log(2.0))
    assert isclose((nxarr * 2.0).log_norm - log_norm_nxarr, log(2.0))
    print("Test scalar multiplication: OK")


def test_eq() -> None:
    arr1 = normal(size=(2, 3, 4))
    nxarr1 = NXArray(arr1, "i1", "i2", "i3")
    arr2 = normal(size=(2, 3, 4))
    nxarr2 = NXArray(arr2, "i1", "i2", "i3")
    _nxarr1 = nxarr1._transpose("i2", "i1", "i3")
    assert _nxarr1 == 1.0 * nxarr1
    assert nxarr1 != nxarr2
    assert nxarr1 != nxarr1 * 2.0
    assert nxarr1 != 2.0 * nxarr1
    print("Test equality: OK")

def test_qr() -> None:
    arr = normal(size=(2, 3, 4, 5))
    nxarr = NXArray(arr, "i1", "i2", "i3", "i4")
    lhs, rhs = linalg.qr(nxarr[..., "i4", "i1"], "?")
    assert lhs.index_ids == ("i2", "i3", "?")
    assert lhs.shape == (3, 4, 10)
    assert rhs.index_ids == ("?", "i4", "i1")
    assert rhs.shape == (10, 5, 2)
    q = reshape(lhs.release_array("i2", "i3", "?"), (-1, 10))
    assert isclose(q.T @ q, eye(10, 10)).all()
    _nxarr = lhs * rhs
    assert _nxarr == nxarr
    assert isclose(_nxarr.release_normalized_array("i1", "i2", "i3", "i4"), nxarr.release_normalized_array("i1", "i2", "i3", "i4")).all()
    assert isclose(_nxarr.log_norm, nxarr.log_norm), f"{_nxarr.log_norm}, {nxarr.log_norm}"
    lhs, rhs = linalg.qr(nxarr["i4", "i1"], "?")
    assert lhs.index_ids == ("i4", "i1", "?")
    assert lhs.shape == (5, 2, 10)
    assert rhs.index_ids == ("?", "i2", "i3")
    assert rhs.shape == (10, 3, 4)
    q = reshape(lhs.release_array("i4", "i1", "?"), (-1, 10))
    assert isclose(q.T @ q, eye(10, 10)).all()
    _nxarr = lhs * rhs
    assert _nxarr == nxarr
    assert isclose(_nxarr.release_normalized_array("i1", "i2", "i3", "i4"), nxarr.release_normalized_array("i1", "i2", "i3", "i4")).all()
    assert isclose(_nxarr.log_norm, nxarr.log_norm), f"{_nxarr.log_norm}, {nxarr.log_norm}"


if __name__ == "__main__":
    test_2_norm()
    test_scalar_mul()
    test_eq()
    test_qr()
