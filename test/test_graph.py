import torch
import glocal_gnn as graph


def test_two_local():
    # Line-graph with 3 nodes.
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    index, assignment, iso_type = graph.two_local(index, 3)
    assert index.max().item() + 1 == 3  # 3 nodes.
    assert index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    assert assignment.tolist() == [[0, 1, 0, 2, 1, 2], [0, 0, 1, 1, 2, 2]]
    assert iso_type.tolist() == [1, 0, 1]


def test_connected_two_local():
    # Line-graph with 3 nodes.
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    index, assignment, iso_type = graph.connected_two_local(index, 3)
    assert index.max().item() + 1 == 2  # 2 nodes.
    assert index.tolist() == [[0, 1], [1, 0]]
    assert assignment.tolist() == [[0, 1, 1, 2], [0, 0, 1, 1]]
    assert iso_type.tolist() == [1, 1]


def test_two_malkin():
    # Line-graph with 3 nodes.
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    index, assignment, iso_type = graph.two_malkin(index, 3)
    assert index.max().item() + 1 == 3  # 3 nodes.
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert assignment.tolist() == [[0, 1, 0, 2, 1, 2], [0, 0, 1, 1, 2, 2]]
    assert iso_type.tolist() == [1, 0, 1]


def test_connected_two_malkin():
    # Triangle-graph with 3 nodes.
    index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])

    index, assignment, iso_type = graph.connected_two_malkin(index, 3)
    assert index.max().item() + 1 == 3  # 3 nodes.
    assert index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    assert assignment.tolist() == [[0, 1, 0, 2, 1, 2], [0, 0, 1, 1, 2, 2]]
    assert iso_type.tolist() == [1, 1, 1]
