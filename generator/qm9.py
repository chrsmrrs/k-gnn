import os
import os.path as osp
from six.moves import urllib
import errno
import tarfile
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
from rdkit.Chem import AllChem  # noqa
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    print('Downloading', url)
    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def extract_tar(path, folder, mode='r:gz', log=True):
    print('Extracting', path)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def coalesce(index, value):
    n = index.max().item() + 1
    row, col = index
    unique, inv = torch.unique(row * n + col, sorted=True, return_inverse=True)

    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]], dim=0)
    value = value[perm]

    return index, value


data_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
           'datasets/gdb9.tar.gz'
# file_path = download_url(data_url, '/Users/rusty1s/Desktop')
# extract_tar(file_path, '/Users/rusty1s/Desktop', mode='r')

suppl = Chem.SDMolSupplier('/Users/rusty1s/Desktop/gdb9.sdf')

with open('/Users/rusty1s/Desktop/gdb9.sdf.csv', 'r') as f:
    target = f.read().split('\n')[1:-1]
    target = [[float(x) for x in line.split(',')[4:16]] for line in target]
    target = torch.tensor(target, dtype=torch.float)

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
data_list = []

for i, mol in enumerate(suppl):
    if mol is None:
        continue

    text = suppl.GetItemText(i)
    num_hs = []
    for atom in mol.GetAtoms():
        num_hs.append(atom.GetTotalNumHs())

    mol = Chem.AddHs(mol)
    feats = factory.GetFeaturesForMol(mol)

    H_type = []
    C_type = []
    N_type = []
    O_type = []
    F_type = []
    atomic_number = []
    sp = []
    sp2 = []
    sp3 = []
    aromatic = []
    acceptor = []
    donor = []

    # AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    # pos = []

    # Example 130669 has an error and yields a different number of atoms.
    # We discard it.
    if i == 130669:
        continue
    # num_atoms = int(text.split('\n')[3].split()[0])
    # if num_atoms != mol.GetNumAtoms():
    #     print('Error at Atom', i)
    #     continue
    num_atoms = mol.GetNumAtoms()

    print(i)
    pos = text.split('\n')[4:4 + num_atoms]
    pos = [[float(x) for x in line.split()[:3]] for line in pos]

    for j in range(num_atoms):
        atom = mol.GetAtomWithIdx(j)
        symbol = atom.GetSymbol()
        H_type.append(1 if symbol == 'H' else 0)
        C_type.append(1 if symbol == 'C' else 0)
        N_type.append(1 if symbol == 'N' else 0)
        O_type.append(1 if symbol == 'O' else 0)
        F_type.append(1 if symbol == 'F' else 0)
        atomic_number.append(atom.GetAtomicNum())
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        acceptor.append(0)
        donor.append(0)

        if symbol == 'H':
            num_hs.insert(j, 0)

        # p = mol.GetConformer().GetAtomPosition(j)
        # pos.append([p.x, p.y, p.z])

    for j in range(0, len(feats)):
        if feats[j].GetFamily() == 'Donor':
            node_list = feats[j].GetAtomIds()
            for j in node_list:
                donor[j] = 1
        elif feats[j].GetFamily() == 'Acceptor':
            node_list = feats[j].GetAtomIds()
            for j in node_list:
                acceptor[j] = 1

    x = [
        H_type, C_type, N_type, O_type, F_type, atomic_number, acceptor, donor,
        aromatic, sp, sp2, sp3, num_hs
    ]
    x = torch.tensor(x, dtype=torch.float).t().contiguous()
    pos = torch.tensor(pos, dtype=torch.float)
    y = target[i].view(1, 12)

    row, col, single, double, triple, aromatic = [], [], [], [], [], []

    # Complete graph
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue

            row.append(i)
            col.append(j)
            e_ij = mol.GetBondBetweenAtoms(i, j)

            if e_ij is not None:
                bond_type = e_ij.GetBondType()
                single.append(1 if bond_type == BondType.SINGLE else 0)
                double.append(1 if bond_type == BondType.DOUBLE else 0)
                triple.append(1 if bond_type == BondType.TRIPLE else 0)
                aromatic.append(1 if bond_type == BondType.AROMATIC else 0)
            else:
                single.append(0)
                double.append(0)
                triple.append(0)
                aromatic.append(0)

    # Non-complete graph
    # for bond in mol.GetBonds():
    #     start = bond.GetBeginAtomIdx()
    #     end = bond.GetEndAtomIdx()

    #     row.append(start)
    #     col.append(end)

    #     row.append(end)
    #     col.append(start)

    #     bond_type = bond.GetBondType()
    #     single.append(1 if bond_type == BondType.SINGLE else 0)
    #     single.append(single[-1])
    #     double.append(1 if bond_type == BondType.DOUBLE else 0)
    #     double.append(double[-1])
    #     triple.append(1 if bond_type == BondType.TRIPLE else 0)
    #     triple.append(triple[-1])
    #     aromatic.append(1 if bond_type == BondType.AROMATIC else 0)
    #     aromatic.append(aromatic[-1])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(
        [single, double, triple, aromatic],
        dtype=torch.float).t().contiguous()

    edge_index, edge_attr = coalesce(edge_index, edge_attr)

    assert pos.size(0) == x.size(0)
    assert edge_index.size(1) == edge_attr.size(0)
    assert edge_index.max().item() + 1 <= x.size(0)

    data_list.append({
        'x': x,
        'y': y,
        'pos': pos,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    })

torch.save(data_list, '/Users/rusty1s/Desktop/qm9.pt')
print(len(data_list))
