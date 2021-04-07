import torch


def do_auction():
    cost = torch.Tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 0]])

    cost_hb = cost.hammerblade()

    print('hi')
    res = torch.auction(cost_hb)

    print('hi again')
    print(res.cpu())


if __name__ == '__main__':
    do_auction()
