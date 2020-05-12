import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_mean
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv
from k_gnn import GraphConv, DataLoader, avg_pool
from k_gnn import ConnectedThreeMalkin
import numpy as np

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6  # Remove graphs with less than 6 nodes.


class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = ConnectedThreeMalkin()(data)
        data.x = x
        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1-23-QM9')
dataset = QM9(
    path,
    transform=T.Compose([T.Distance(norm=False)]),
    pre_transform=MyPreTransform(),
    pre_filter=MyFilter())
dataset.data.y = dataset.data.y[:,0:12]

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = F.one_hot(
    dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)





class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        M_in, M_out = dataset.num_features, 32
        nn1 = Sequential(Linear(6, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(6, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(6, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(data.x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=5, min_lr=0.00001)




results = []
for _ in range(5):


    dataset = dataset.shuffle()


    tenpercent = int(len(dataset) * 0.1)
    print("###")
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    print("###")
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent:].shuffle()

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # TODO
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                           min_lr=0.0000001)

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))


    def test(loader):
        model.eval()
        error = 0

        lf = torch.nn.L1Loss()

        for data in loader:
            data = data.to(device)
            error += lf(model(data) * std, data.y * std).item()
        return error / len(loader.dataset)


    best_val_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())