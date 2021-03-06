"""
Tests on torch.mm
03/10/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import random
import pytest

torch.manual_seed(42)
random.seed(42)

def test_torch_mm_1():
    mat1 = torch.ones(2, 3)
    mat2 = torch.ones(3, 3)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1, mat2)
    out_h = torch.mm(mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)

def test_torch_mm_2():
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1, mat2)
    out_h = torch.mm(mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)

@pytest.mark.xfail
def test_torch_mm_mismatching_shape_F():
    mat1 = torch.randn(2, 2)
    mat2 = torch.randn(3, 3)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out_h = torch.mm(mat1_h, mat2_h)

def test_torch_mm_transpose_1():
    mat1 = torch.randn(3, 4)
    mat2 = torch.randn(3, 5)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1.t(), mat2)
    out_h = torch.mm(mat1_h.t(), mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)

def test_torch_mm_large():
    mat1 = torch.randn(64, 19)
    mat2 = torch.randn(19, 32)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1, mat2)
    out_h = torch.mm(mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out, rtol=1e-04, atol=1e-06)
