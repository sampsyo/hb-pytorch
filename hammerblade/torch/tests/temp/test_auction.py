import torch


def do_auction():
    cost = torch.Tensor([[8,   7, 6],
                         [100, 1, 2],
                         [1,   2, 3]])

    print('123 hi')
    cost_hb = cost.hammerblade()

    print('hi')
    res = torch.auction(cost_hb)

    print('hi again')
    print(res.cpu())


if __name__ == '__main__':
    do_auction()
