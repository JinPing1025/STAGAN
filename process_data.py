
from rdkit import Chem

def data_to_mol(data):
    data_mol = []
    data_none = []
    for i in range(len(data)):
        mol = data[i]
        m = Chem.MolFromSmiles(mol)
        if m is None:
            data_none.append(m)
            print(i)
        else:
            data_mol.append(m)
    return data_mol,data_none

def smiles_to_mol(smiles, catch_errors=True):
    """Generates RDKit Mol object from a SMILES string"""
    if catch_errors:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        flag = Chem.SanitizeMol(mol, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)
            return mol
    return Chem.MolFromSmiles(smiles, sanitize=True)

def atom_nums(data):
    atom_num=[]
    for i in range(len(data)):
        molecule = smiles_to_mol(data[i])
        a = molecule.GetNumHeavyAtoms()
        atom_num.append(a)
    return atom_num

def atom_dictionarys(atom_num):
    atom_dictionary ={}
    for i in atom_num:
        atom_dictionary[i] = atom_num.count(i)
    return atom_dictionary

#选择合适重原子个数的小分子
def elect_data(data,nums):
    smile_data=[]
    for i in range(len(data)):
        molecule = smiles_to_mol(data[i])
        if molecule.GetNumHeavyAtoms()==nums:
            smile = data[i]
            smile_data.append(smile)
    return smile_data

#转化分子为kekule式
def kekule_mol(data):
    train_data = []
    for i in data:
        mol = Chem.MolFromSmiles(i, sanitize=False)
        Chem.Kekulize(mol)
        molecule = Chem.MolToSmiles(mol,isomericSmiles=False,kekuleSmiles=True)
        train_data.append(molecule)
    return train_data

# def delet_element(data):
#     dataset = []
#     delet_smile = []
#     for i in range(len(data)):
#         if '[O-]' in data[i] or 'Br' in data[i] or 'I' in data[i] or \
#             'P' in data[i] or '[NH]' in data[i] or '[N+]' in data[i] or \
#                 'S' in data[i] or 'B' in data[i] or \
#                     'Sn' in data[i] or 'Si' in data[i]:
#                         print(i)
#                         delet_smile.append(data[i])
#         else:
#             dataset.append(data[i])
#     return dataset ,delet_smile

#GDB17数据集专用
def delet_element(data):
    dataset = []
    delet_smile = []
    for i in range(len(data)):
        if '[O-]' in data[i] or 'Br' in data[i] or 'I' in data[i] or \
            'P' in data[i] or '[NH]' in data[i] or '[N+]' in data[i] or \
                'Cl' in data[i] or 'B' in data[i] or 'S' in data[i] or\
                    'Sn' in data[i] or 'Si' in data[i] or '[nH]' in data[i] or\
                        '[NH3+]' in data[i] or '[NH]' in data[i] or '[N-]' in data[i] or\
                            '[CH-]' in data[i] or '[NH2+]' in data[i] or '[C-]' in data[i]:
                                print(i)
                                delet_smile.append(data[i])
        else:
            dataset.append(data[i])
    return dataset ,delet_smile
