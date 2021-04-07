import torch


def do_auction():
    out = torch.Tensor([0, 0, 0])
    cost = torch.Tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 0]])

    out.hammerblade()
    cost.hammerblade()
    print(torch.auction(out, cost))


if __name__ == '__main__':
    do_auction()
