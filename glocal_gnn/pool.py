from torch_scatter import scatter_add, scatter_max, scatter_mean


def add_pool(x, assignment):
    row, col = assignment
    return scatter_add(x[row], col, dim=0)


def max_pool(x, assignment):
    row, col = assignment
    return scatter_max(x[row], col, dim=0)[0]


def avg_pool(x, assignment):
    row, col = assignment
    return scatter_mean(x[row], col, dim=0)
