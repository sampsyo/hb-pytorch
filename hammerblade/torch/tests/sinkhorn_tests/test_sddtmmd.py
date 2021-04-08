"""
Unit tests for validating sddtmmd kernel
08/28/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def sddmm_expected(a, b, c):
    outvals = torch.zeros(a._nnz())
    for k in range(a._nnz()):
        ai, aj = tuple(a._indices()[:, k].tolist())
        brow = b[ai, :]
        ccol = c[:, aj]
        outvals[k] = torch.dot(brow, ccol)
    return torch.sparse.FloatTensor(
        a._indices(),
        outvals,
        a.shape,
    ).to_dense()

def _test_torch_sddtmmd(a, b, c):
    expected_tensor = sddmm_expected(a, b, c.t())
    ah = a.hammerblade()
    bh = b.hammerblade()
    ch = c.hammerblade()
    got_hb = torch.sddtmmd(ah, bh, ch)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    # compare HB with calculated sddtmmd in python
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)
    # compare HB with CPU
    expected_tensor_cpu = torch.sddtmmd(a, b, c)
    assert torch.equal(got_tensor, expected_tensor_cpu)


def test_torch_sddtmmd_1():
    a = torch.Tensor([[1, 0, 1], [0, 3, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2, 1], [2, 1, 1]]).t()
    _test_torch_sddtmmd(a, b, c)

def test_torch_sddtmmd_2():
    a = torch.Tensor([[0, 0], [0, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1]]).t()
    _test_torch_sddtmmd(a, b, c)

def test_torch_sddtmmd_3():
    a = torch.Tensor([[1, 0], [0, 1], [2, 0]]).to_sparse()
    b = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    c = torch.Tensor([[1, 1], [2, 1], [3, 2]]).t()
    _test_torch_sddtmmd(a, b, c)
