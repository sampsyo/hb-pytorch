import numpy
import scipy.sparse
import os
import sys
import torch
import json

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils import parse_model_args, train, inference, save_model  # noqa
from time import time

# Kernel parameters.
TOTAL_DOCS = 4096
QUERY_IDX = 5  # Was 100; lowered to allow even smaller runs.
LAMBDA = 1

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

# Kernel "routing" file.
ROUTE_JSON = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd.json')


def swmd_torch(r, cT, vecs, niters):
    """The actual Sinkhorn WMD kernel.
    """
    # I=(r > 0)
    sel = r > 0

    # r=r(I)
    r = r[sel].reshape(-1, 1)

    # M=M(I,:)
    M = torch.cdist(vecs[sel], vecs)

    # x=ones(length(r), size(c,2)) / length(r)
    a_dim = r.shape[0]
    b_nobs = cT.shape[0]
    xT = torch.ones((b_nobs, a_dim)) / a_dim

    # K=exp(-lambda * M)
    K = torch.exp(- M * LAMBDA)
    K_div_r = K / r
    K_T = K.T

    # BEGIN PROFILING HERE
    start_time = time()
    # torch.hammerblade.profiler.route.set_route_from_json(data)
    torch.hammerblade.profiler.enable()

    for it in range(niters):
        print('starting iteration {}'.format(it))

        uT = 1.0 / xT

        # Interesting property: sddtmmt(a,b,c) = sddtmm(a.T,c,b)
        # Compute `c * 1/(K_T @ u)` using a hand-rolled SDDMM.
        # v = c * (1.0 / _sddmm(c, K_T, u))
        # v = c * (1.0 / torch.sddtmm(c, K_T, uT)
        # vT = cT * torch.sddtmm(cT, uT, K_T).sparse_reciprocal()

        # NOTE: NEED TO ADD RECIPROCAL
        vT = cT * torch.sddtmm(cT, uT, K_T)

        # custom dstmm.t():
        # x = _dsmp(K_div_r, v)
        # x = torch.dstmm(K_div_r, vT)
        xT = torch.dstmmt(K_div_r, vT)

    out = (uT.t() * torch.dstmm(K * M, vT)).sum(axis=0)

    # out = (uT * (vT @ (K_T * M.t())).sum(axis=1) 
    # Note: M is huge compared to uT, so use the sum(axis=0) instead of sum(axis=1) line

    # END PROFILING HERE
    torch.hammerblade.profiler.disable()
    end_time = time()
    elapsed = end_time - start_time
    print("elapsed:", elapsed)
    print("elapsed * 16:", elapsed * 16)

    return out


def load_data(n_docs):
    """Load data for the Sinkhorn WMD kernel.
    """
    # Load data.
    vecs = numpy.load(DATA_VECS)
    mat = scipy.sparse.load_npz(DATA_MAT)
    print("vecs size:", vecs.shape)
    mat = mat[:, :n_docs]  # Use a subset of the data.
    print("mat shape:", mat.shape)
    # The query vector.
    r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()

    # mat could theoretically be stored as its transpose, so don't count 
    matT = mat.T

    # Convert arrays to PyTorch tensors.
    r = torch.FloatTensor(r)
    cT_coo = matT.tocoo()
    cT = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((cT_coo.row, cT_coo.col))),
        torch.FloatTensor(cT_coo.data),
        torch.Size(cT_coo.shape),
    )

    vecs = torch.FloatTensor(vecs)

    return r, cT, vecs


def sinkhorn_test():
    # Use `--hb` to run in HammerBlade mode. Otherwise, we run all native.
    # Optionally add a number to offload only a specific kernel.
    args = sys.argv[1:]
    if '--hb' in args:
        on_hb = True
        args.remove('--hb')
        if args:
            # The index of the specific kernel to offload.
            kernel_idx = int(args[0])
        else:
            kernel_idx = None
    else:
        on_hb = False

    # Set up HammerBlade cosim stuff.
    if on_hb:
        torch.hammerblade.init()

        # Set up HammerBlade "routing," which tells kernels to run on HB
        # instead of on the CPU.
        if on_hb:
            with open(ROUTE_JSON) as f:
                route_data = json.load(f)
            for i, kernel in enumerate(route_data):
                # Mark kernel for offload.
                if kernel_idx is None or kernel_idx == i:
                    print('offloading kernel', kernel['signature'])
                    kernel['offload'] = True

                # Set up a "chart" "beacon" (?).
                torch.hammerblade.profiler.chart.add(kernel['signature'])

            torch.hammerblade.profiler.route.set_route_from_json(route_data)

    # Set the size of the run. Use TOTAL_DOCS/data_fraction of the data.
    data_fraction = 16 if on_hb else 1  # Subset on HB.
    n_docs = TOTAL_DOCS // data_fraction

    # Load data and run the kernel.
    print('loading data for {} docs'.format(n_docs))
    r, cT, vecs = load_data(n_docs)
    print('done loading data; running kernel')
    scores = swmd_torch(r, cT, vecs, niters=1)

    # Dump profiling results, including both the overall statistics and the
    # invocation "tree" that breaks down every call stack.
    print(torch.hammerblade.profiler.stats(trimming=True))
    print(torch.hammerblade.profiler.exec_time.raw_stack())

    print("done")
    print("Multiply sddtmm, dstmmt, and dstmm times by",
          data_fraction, "for true time on real dataset.")


if __name__ == '__main__':
    sinkhorn_test()
