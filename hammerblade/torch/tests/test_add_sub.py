"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# test of adding two tensors

def _test_add(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 + x2
    y_h = h1 + h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # # inplace
    # x1.add_(x2)
    # h1.add_(h2)
    # assert h1.device == torch.device("hammerblade")
    # assert torch.allclose(x1, h1.cpu())

def test_add_8():
    x1 = torch.rand(8)
    x2 = torch.rand(8)
    _test_add(x1, x2)

def test_add_64():
    x1 = torch.rand(64)
    x2 = torch.rand(64)
    _test_add(x1, x2)

def test_add_128():
    x1 = torch.rand(128)
    x2 = torch.rand(128)
    _test_add(x1, x2)

def test_add_256():
    x1 = torch.rand(256)
    x2 = torch.rand(256)
    _test_add(x1, x2)

def test_add_512():
    x1 = torch.rand(512)
    x2 = torch.rand(512)
    _test_add(x1, x2)

def test_add_1024():
    x1 = torch.rand(1024)
    x2 = torch.rand(1024)
    _test_add(x1, x2)

def test_add_2048():
    x1 = torch.rand(2048)
    x2 = torch.rand(2048)
    _test_add(x1, x2)

def test_add_4096():
    x1 = torch.rand(4096)
    x2 = torch.rand(4096)
    _test_add(x1, x2)

