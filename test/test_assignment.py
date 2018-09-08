import torch
import graph_cpu


def test_assignment_2to_3():
    # Line-graph with 4 nodes.
    index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    assignment = graph_cpu.assignment_2to3(index, 4)
    assert assignment.tolist() == [[0, 1, 3, 3, 4, 5], [0, 0, 0, 1, 1, 1]]
