import torch


def do_auction():
    cost = torch.Tensor([
        [8, 7, 6, 1, 2, 3, 4, 5],
        [9, 1, 2, 1, 2, 3, 4, 5],
        [1, 2, 3, 1, 2, 3, 4, 5],
        [7, 6, 1, 2, 8, 3, 4, 5],
        [1, 2, 1, 2, 9, 3, 4, 5],
        [2, 3, 1, 2, 1, 3, 4, 5],
        [1, 2, 1, 2, 3, 4, 9, 5],
        [7, 1, 6, 2, 8, 3, 4, 5],
    ])

    print('123 hi')
    cost_hb = cost.hammerblade()

    print('hi')
    res = torch.auction(cost_hb)

    print('hi again')
    print(res.cpu())


if __name__ == '__main__':
    do_auction()
