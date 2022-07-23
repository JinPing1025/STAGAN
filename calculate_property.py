
'''valid, novelty, unique'''
def check_valid(generator_molecules):
    valid_molecules=[]
    molecules = list(filter(lambda x: x is not None, generator_molecules))
    for i in  molecules:
        s = Chem.MolToSmiles(i) 
        if '*' not in s  and '.' not in s and s != '':
            valid_molecules.append(s)
    return valid_molecules


def valid_lambda(x):
    return x is not None and Chem.MolToSmiles(x) != ''

def valid_filter(mols):
    return list(filter(valid_lambda, mols))

def mol_to_smiles(mols):
    return Chem.MolToSmiles(mols)
    
def check_novelty(generator_molecules, data):
    v = valid_filter(generator_molecules)
    novel = list(filter(lambda x: valid_lambda(x) and Chem.MolToSmiles(x) not in data, v))
    novel_molecules = list(map(mol_to_smiles,novel))
    return novel_molecules

def check_unique(generator_molecules):
    a = list(filter(valid_lambda,generator_molecules))
    unique_molecules = set(map(lambda x: Chem.MolToSmiles(x), a))
    return unique_molecules


'''QED'''
from rdkit import Chem
from rdkit.Chem.QED import qed

def calculate_qed(dataset):
    dataset_qed=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = qed(m)
        dataset_qed.append(a)
    return dataset_qed

'''SA_score'''
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.Draw import IPythonConsole
import sascorer
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def calculate_SA(dataset):
    generator_SA = pd.DataFrame(dataset,columns=['smiles'])
    PandasTools.AddMoleculeColumnToFrame(frame=generator_SA, smilesCol='smiles')
    generator_SA['calc_SA_score'] = generator_SA.ROMol.map(sascorer.calculateScore)
    generator_SA_score = generator_SA.calc_SA_score
    return generator_SA,generator_SA_score

def draw_SA_molecule(data_SA_score,data_SA):
    (id_max, id_min) = (data_SA_score.idxmax(), data_SA_score.idxmin())
    sa_mols = [data_SA.ROMol[id_max],data_SA.ROMol[id_min]]
    img = Draw.MolsToGridImage(sa_mols, subImgSize=(340,200),
                         legends=['SA-score: {:.2f}'.format(data_SA.calc_SA_score[i]) for i in [id_max, id_min]])
    return img


from rdkit.Chem  import Descriptors

'''Molecular weight (MW)'''
def calculate_MW(dataset):
    dataset_MW=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Descriptors.MolWt(m)
        dataset_MW.append(a)
    return dataset_MW

'''LogP'''
def calculate_LogP(dataset):
    dataset_LogP=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Descriptors.MolLogP(m)
        dataset_LogP.append(a)
    return dataset_LogP

''' five rule of Lipinski  
    MW < 500, LogP < 5, NumHDonors < 5, NumHAcceptors < 10, NumRotatableBonds <= 10'''

from rdkit import Chem
from rdkit.Chem import Lipinski

'''NumHDonors'''
def calculate_NumHDonors(dataset):
    dataset_NumHDonors=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumHDonors(m)
        dataset_NumHDonors.append(a)
    return dataset_NumHDonors

'''NumHAcceptors'''
def calculate_NumHAcceptors(dataset):
    dataset_NumHAcceptors=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumHAcceptors(m)
        dataset_NumHAcceptors.append(a)
    return dataset_NumHAcceptors

'''NumRotatableBonds'''
def calculate_NumRotatableBonds(dataset):
    dataset_NumRotatableBonds=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumRotatableBonds(m)
        dataset_NumRotatableBonds.append(a)
    return dataset_NumRotatableBonds


'''核密度函数分布'''
import seaborn as sns

def density_estimation(password,generator_data,train_data):
    sns.set(color_codes=True)
    sns.set_style("white")
    if password == 'qed_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_qed"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"data_qed"},color="r")
    elif password == 'SA_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_SA_score"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_SA_score"},color="r")
    elif password == 'MW_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_MW"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_MW"},color="r")
    elif password == 'LogP_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_LogP"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_LogP"},color="r")
    elif password == 'NumHDonors_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_NumHDonors"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_NumHDonors"},color="r")
    elif password == 'NumHAcceptors_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_NumHAcceptors"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_NumHAcceptors"},color="r")
    elif password == 'NumRotatableBonds_value':
        ax1 = sns.distplot(generator_data,hist=False,kde_kws={"label":"generator_NumRotatableBonds"},color="b")
        ax2 = sns.distplot(train_data,hist=False,kde_kws={"label":"train_NumRotatableBonds"},color="r")
    return ax1,ax2
    
