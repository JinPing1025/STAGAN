'''
If you want to calculate these properties, you just make a selection call, 
depending on the variables of the function
'''

'''Properties distribution'''
def remove_none(dataset):
    generator_data=[]
    generator_mol=[]
    generator_none=[]
    for i in range(len(dataset)):
        mol = dataset[i]
        m = Chem.MolFromSmiles(mol)
        if m is None:
            generator_none.append(m)
            print(i)
        else:
            generator_data.append(dataset[i])
            generator_mol.append(m)
    return generator_data,generator_mol

generator_data,generator_mol = remove_none(valid_molecules)

'''Fréchet ChemNet Distance'''
from Metrics import FCD_score

fcd_score = FCD_score(generator_data,data,sampel_size=5000)
print('FCD: ',fcd_score)


from calculate_property import calculate_qed,calculate_SA
from calculate_property import density_estimation
from calculate_property import draw_SA_molecule, calculate_MW, calculate_LogP

'''QED'''
generator_qed = calculate_qed(generator_data)
data_qed = calculate_qed(data)

password='qed_value'
ax1,ax2 = density_estimation(password,generator_qed,data_qed)

'''SA_score'''
generator_SA,generator_SA_score = calculate_SA(generator_data)
data_SA,data_SA_score = calculate_SA(data)

img1=draw_SA_molecule(generator_SA_score,generator_SA)
img1
img2=draw_SA_molecule(data_SA_score,data_SA)
img2

password='SA_value'
ax1,ax2 = density_estimation(password,generator_SA_score,data_SA_score)

'''Molecular weight (MW)'''
generator_MW = calculate_MW(generator_data)
data_MW = calculate_MW(data)

password='MW_value'
ax1,ax2 = density_estimation(password,generator_MW,data_MW)

'''LogP'''
generator_LogP = calculate_LogP(generator_data)
data_LogP = calculate_LogP(data)

password='LogP_value'
ax1,ax2 = density_estimation(password,generator_LogP,data_LogP)

"""
five rule of Lipinski  
MW < 500, LogP < 5, NumHDonors < 5, NumHAcceptors < 10, NumRotatableBonds <= 10
"""

from calculate_property import calculate_NumHDonors
from calculate_property import calculate_NumHAcceptors
from calculate_property import calculate_NumRotatableBonds

'''NumHDonors'''
generator_NumHDonors = calculate_NumHDonors(generator_data)
data_NumHDonors = calculate_NumHDonors(data)

password='NumHDonors_value'
ax1,ax2 = density_estimation(password,generator_NumHDonors,data_NumHDonors)

'''NumHAcceptors'''
generator_NumHAcceptors = calculate_NumHAcceptors(generator_data)
data_NumHAcceptors = calculate_NumHAcceptors(data)

password='NumHAcceptors_value'
ax1,ax2 = density_estimation(password,generator_NumHAcceptors,data_NumHAcceptors)

'''NumRotatableBonds'''
generator_NumRotatableBonds = calculate_NumRotatableBonds(generator_data)
data_NumRotatableBonds = calculate_NumRotatableBonds(data)

password='NumRotatableBonds_value'
ax1,ax2 = density_estimation(password,generator_NumRotatableBonds,data_NumRotatableBonds)

'''Common Metrics'''
'''internal_diversity'''
from Metrics import fingerprint
from Metrics import internal_diversity

generator_Fps = fingerprint(generator_mol,fp_type='morgan')
data_Fps = fingerprint(data_mol,fp_type='morgan')

generator_IntDiv1 = internal_diversity(generator_Fps,p=1)
generator_IntDiv2 = internal_diversity(generator_Fps,p=2)

data_IntDiv1 = internal_diversity(data_Fps,p=1)
data_IntDiv2 = internal_diversity(data_Fps,p=2)

print(generator_IntDiv1)
print(generator_IntDiv2)
print(data_IntDiv1)
print(data_IntDiv2)


'''Similarity to a nearest neighbor'''
from Metrics import SNN

SNN_score = SNN(generator_Fps,data_Fps,p=1)
print('SNN: ',SNN_score)


'''Fréchet ChemNet Distance'''
from Metrics import FCD_score

fcd_score = FCD_score(generator_data,data,sampel_size=6000)
print('FCD: ',fcd_score)


'''Fragment similarity'''
from Metrics import compute_fragments
from Metrics import cos_similarity

generator_frag = compute_fragments(generator_mol, n_jobs=1)
data_frag = compute_fragments(data_mol, n_jobs=1)

Frag_score = cos_similarity(data_frag, generator_frag)
print('Frag: ',Frag_score)

'''Scaffold similarity (Scaff)'''
from Metrics import compute_scaffolds

generator_scaf = compute_scaffolds(generator_mol, n_jobs=1, min_rings=2)
data_scaf = compute_scaffolds(data_mol, n_jobs=1, min_rings=2)

Scaf_score = cos_similarity(data_scaf, generator_scaf)
print('Scaf: ',Scaf_score)
