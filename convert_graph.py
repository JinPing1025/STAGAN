
from rdkit import Chem
from process_data import smiles_to_mol

import numpy as np


# atom_mapping = {"C": 0,0: "C","N": 1,1: "N","O": 2,2: "O","F": 3,3: "F","Cl":4,4:"Cl"}
atom_mapping = {"C": 0,0: "C","N": 1,1: "N","O": 2,2: "O","F": 3,3: "F"}

bond_mapping = {"SINGLE": 0, 0: Chem.BondType.SINGLE,"DOUBLE": 1, 1: Chem.BondType.DOUBLE,
                "TRIPLE": 2, 2: Chem.BondType.TRIPLE,"AROMATIC": 3, 3: Chem.BondType.AROMATIC}

def smiles_to_graph(smiles,num_atoms,atom_dim,bond_dim):
    #molecule = Chem.MolFromSmiles(smiles)
    molecule=smiles_to_mol(smiles,catch_errors=True)
    
    adjacency = np.zeros((bond_dim, num_atoms, num_atoms), "float32")
    features = np.zeros((num_atoms, atom_dim), "float32")
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(atom_dim)[atom_type]    #生成对角阵,主对角线上元素为1,其余位置均为0
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1      
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1
    return adjacency, features

def graph_to_molecule(graph,num_atoms,atom_dim,bond_dim):
    adjacency, features = graph
    molecule = Chem.RWMol()
    #删除无原子,无键原子
    keep_idx = np.where((np.argmax(features, axis=1) != atom_dim - 1)&(np.sum(adjacency[:-1], axis=(0, 1)) != 0))[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]
    #添加原子到分子中
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        a = molecule.AddAtom(atom)
    #添加键到分子
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == bond_dim - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)
    # flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # if flag != Chem.SanitizeFlags.SANITIZE_NONE:
    #     return None
    return molecule

def convert_tensor(data,num_atoms,atom_dim,bond_dim):
    adjacency = np.zeros((bond_dim, num_atoms, num_atoms), "float32")
    features = np.zeros((num_atoms, atom_dim), "float32")
    adjacency_tensor = []
    feature_tensor = []
    for smiles in data:
        adjacency, features = smiles_to_graph(smiles,num_atoms,atom_dim,bond_dim)
        adjacency_tensor.append(adjacency)
        feature_tensor.append(features)
    adjacency_tensor = np.array(adjacency_tensor)
    feature_tensor = np.array(feature_tensor)
    return adjacency_tensor,feature_tensor
