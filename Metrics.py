
'''average Tanimoto similarity'''
import torch
import numpy as np

def average_agg_tanimoto(stock_vecs, gen_vecs,batch_size=5000, agg='max',device='cpu', p=1):
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


'''fingerprint'''
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def fingerprint(dataset,fp_type='maccs'):
    if fp_type == 'maccs':
        Fps = [MACCSkeys.GenMACCSKeys(x) for x in dataset]
    elif fp_type == 'morgan':
        Fps =[AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024,useFeatures=True) for x in dataset] 
        Fps = np.asarray(Fps)
    return Fps

'''internal_diversity'''
def internal_diversity(fps,agg='mean',device='cpu', p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    IntDivp = 1 - (average_agg_tanimoto(fps,fps,agg='mean',device=device,p=p)).mean()
    return IntDivp

'''Similarity to a nearest neighbor'''
def SNN(gen_fps,train_fps,agg='max',device='cpu', p=1):
    snn = average_agg_tanimoto(gen_fps,train_fps,agg='max',device=device,p=p)
    return snn
    

'''Fréchet ChemNet Distance'''
import numpy as np
import pandas as pd
import pickle

from fcd import get_fcd, load_ref_model
from fcd import get_predictions, calculate_frechet_distance

def FCD_score(generator_data,train_data,sampel_size=1000):
    """
    输入的smiles分子需要进行预处理，确定不存在None值
    """
    np.random.seed(0)
    # Load chemnet model
    model = load_ref_model()
    sample1 = np.random.choice(generator_data,sampel_size, replace=False)
    sample2 = np.random.choice(train_data,sampel_size, replace=False)
    #get CHEBMLNET activations of generated molecules 
    act1 = get_predictions(model, sample1)
    act2 = get_predictions(model, sample2)
    
    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1.T)
    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2.T)
    fcd_score = calculate_frechet_distance(mu1=mu1,mu2=mu2,
                                           sigma1=sigma1,sigma2=sigma2)
    return fcd_score

'''cos_similarity'''
from scipy.spatial.distance import cosine as cos_distance

def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


'''Fragment similarity'''
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from multiprocessing import Pool

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(mol)
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi

def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments

'''Scaffold similarity (Scaff)'''
from rdkit.Chem.Scaffolds import MurckoScaffold
from functools import partial

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def compute_scaffold(mol, min_rings=2):
    mol = mol
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds
