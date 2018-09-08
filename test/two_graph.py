import torch

import graph_cpu
from glocal_gnn import global_two_graph, local_two_graph


def test_two_graph_idx():
    assert graph_cpu.two_graph_idx(0, 1, 4) == 0
    assert graph_cpu.two_graph_idx(0, 2, 4) == 1
    assert graph_cpu.two_graph_idx(0, 3, 4) == 2
    assert graph_cpu.two_graph_idx(1, 2, 4) == 3
    assert graph_cpu.two_graph_idx(1, 3, 4) == 4
    assert graph_cpu.two_graph_idx(2, 3, 4) == 5

    assert graph_cpu.two_graph_idx(1, 0, 4) == 0
    assert graph_cpu.two_graph_idx(2, 0, 4) == 1
    assert graph_cpu.two_graph_idx(3, 0, 4) == 2
    assert graph_cpu.two_graph_idx(2, 1, 4) == 3
    assert graph_cpu.two_graph_idx(3, 1, 4) == 4
    assert graph_cpu.two_graph_idx(3, 2, 4) == 5


def test_global_two_graph():
    # Line graph with 3 nodes.
    row = torch.tensor([0, 1, 1, 2])
    col = torch.tensor([1, 0, 2, 1])

    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = global_two_graph(edge_index)
    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    assert assignment.tolist() == [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 2]]


def test_local_two_graph():
    # Line graph with 4 nodes.
    row = torch.tensor([0, 1, 1, 2, 2, 3])
    col = torch.tensor([1, 0, 2, 1, 3, 2])
    edge_index = torch.stack([row, col], dim=0)

    edge_index, assignment = local_two_graph(edge_index)
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert assignment.tolist() == [[0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2]]
