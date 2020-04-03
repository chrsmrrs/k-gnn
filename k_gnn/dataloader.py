import torch.utils.data
from torch_geometric.data import Batch


def collate(data_list):
    keys = data_list[0].keys
    assert 'batch' not in keys

    batch = Batch()
    for key in keys:
        batch[key] = []
    batch.batch = []
    if 'edge_index_2' in keys:
        batch.batch_2 = []
    if 'edge_index_3' in keys:
        batch.batch_3 = []

    keys.remove('edge_index')
    props = [
        'edge_index_2', 'assignment_index_2', 'edge_index_3',
        'assignment_index_3', 'assignment_index_2to3'
    ]
    keys = [x for x in keys if x not in props]

    cumsum_1 = N_1 = cumsum_2 = N_2 = cumsum_3 = N_3 = 0

    for i, data in enumerate(data_list):
        for key in keys:
            batch[key].append(data[key])

        N_1 = data.num_nodes
        batch.edge_index.append(data.edge_index + cumsum_1)
        batch.batch.append(torch.full((N_1, ), i, dtype=torch.long))

        if 'edge_index_2' in data:
            N_2 = data.assignment_index_2[1].max().item() + 1
            batch.edge_index_2.append(data.edge_index_2 + cumsum_2)
            batch.assignment_index_2.append(
                data.assignment_index_2 +
                torch.tensor([[cumsum_1], [cumsum_2]]))
            batch.batch_2.append(torch.full((N_2, ), i, dtype=torch.long))

        if 'edge_index_3' in data:
            N_3 = data.assignment_index_3[1].max().item() + 1
            batch.edge_index_3.append(data.edge_index_3 + cumsum_3)
            batch.assignment_index_3.append(
                data.assignment_index_3 +
                torch.tensor([[cumsum_1], [cumsum_3]]))
            batch.batch_3.append(torch.full((N_3, ), i, dtype=torch.long))

        if 'assignment_index_2to3' in data:
            assert 'edge_index_2' in data and 'edge_index_3' in data
            batch.assignment_index_2to3.append(
                data.assignment_index_2to3 +
                torch.tensor([[cumsum_2], [cumsum_3]]))

        cumsum_1 += N_1
        cumsum_2 += N_2
        cumsum_3 += N_3

    keys = [x for x in batch.keys if x not in ['batch', 'batch_2', 'batch_3']]
    for key in keys:
        if torch.is_tensor(batch[key][0]):
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

    batch.batch = torch.cat(batch.batch, dim=-1)

    if 'batch_2' in batch:
        batch.batch_2 = torch.cat(batch.batch_2, dim=-1)

    if 'batch_3' in batch:
        batch.batch_3 = torch.cat(batch.batch_3, dim=-1)

    return batch.contiguous()


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)
