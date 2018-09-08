import torch

from glocal_gnn import add_pool


def test_add_pool():
    assignment = torch.tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2]])

    x = torch.tensor([1, 2, 3, 4])
    assert add_pool(x, assignment).tolist() == [3, 5, 7]

    x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert add_pool(x, assignment).tolist() == [[4, 6], [8, 10], [12, 14]]
