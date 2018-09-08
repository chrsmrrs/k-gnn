import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import one_hot
import torch_geometric.transforms as T
from glocal_gnn import DataLoader, GraphConv, avg_pool
from glocal_gnn import TwoMalkin, ConnectedThreeMalkin

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DD')
dataset = TUDataset(
    path,
    name='DD',
    pre_transform=T.Compose([TwoMalkin(), ConnectedThreeMalkin()]))

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = dataset.data.iso_type_2.max().item() + 1
dataset.data.iso_type_2 = one_hot(dataset.data.iso_type_2, num_classes=num_i_2)

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = one_hot(dataset.data.iso_type_3, num_classes=num_i_3)

dataset = dataset.shuffle()

print(dataset[0])
raise NotImplementedError

n = len(dataset) // 10
dataset = dataset.shuffle()
val_dataset = dataset[:n]
test_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64, 64)
        self.conv5 = GraphConv(64, 64)
        self.conv6 = GraphConv(64, 64)
        self.conv7 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(3 * 64, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(data.x, data.assignment_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(data.x, data.assignment_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=10, min_lr=0.0001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss
        optimizer.step()
    return loss_all / len(train_dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum')
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss_train = train(epoch)
    loss_val = val(val_loader)
    scheduler.step(loss_val)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val Loss: {:.7f}, '
          'Test Acc: {:.7f}'.format(epoch, lr, loss_train, loss_val, test_acc))
