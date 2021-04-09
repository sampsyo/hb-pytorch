import torch


def do_auction():
    cost = torch.Tensor([[8, 2, 3],
                         [4, 1, 6],
                         [7, 8, 1]])

    print('123 hi')
    cost_hb = cost.hammerblade()

    print('hi')
    res = torch.auction(cost_hb)

    print('hi again')
    print(res.cpu())


if __name__ == '__main__':
    do_auction()
