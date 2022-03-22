import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from k_gnn import DataLoader, GraphConv, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeLocal

parser = argparse.ArgumentParser()
parser.add_argument('--no-train', default=False)
args = parser.parse_args()


class MyFilter(object):
    def __call__(self, data):
        return True


class MyPreTransform(object):
    def __call__(self, data):
        return data


BATCH = 32
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', '1-2-3-MUTAG')
dataset = TUDataset(
    path,
    name='MUTAG',
    pre_transform=T.Compose(
        [MyPreTransform(),
         TwoMalkin(), ConnectedThreeLocal()]),
    pre_filter=MyFilter())

perm = torch.randperm(len(dataset), dtype=torch.long)
torch.save(perm, 'mutag_perm.pt')
perm = torch.load('mutag_perm.pt')
dataset = dataset[perm]

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = dataset.data.iso_type_2.max().item() + 1
dataset.data.iso_type_2 = F.one_hot(
    dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = F.one_hot(
    dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 64)
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64 + num_i_2, 64)
        self.conv5 = GraphConv(64, 64)
        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = data.x
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        if args.no_train:
            x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)


def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


acc = []
for i in range(10):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=15, min_lr=0.00001)

    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    n = len(dataset) // 10
    test_mask[i * n:(i + 1) * n] = 1
    test_dataset = dataset[test_mask]
    train_dataset = dataset[~test_mask]

    n = len(train_dataset) // 10
    val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    val_mask[i * n:(i + 1) * n] = 1
    val_dataset = train_dataset[val_mask]
    train_dataset = train_dataset[~val_mask]

    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    print('---------------- Split {} ----------------'.format(i))

    best_val_loss, test_acc = 100, 0
    for epoch in range(1, 51):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        # scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            test_acc = test(test_loader)
            best_val_loss = val_loss
        if epoch % 5 == 0:
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                  'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                      epoch, lr, train_loss, val_loss, test_acc))
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
