

import pandas as pd
import os
#避免运行model.fit或类似代码来开始训练定义好的模型, 服务会挂掉
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


from process_data import smiles_to_mol
from process_data import atom_nums,atom_dictionarys 
from process_data import elect_data

dataset=pd.read_csv(r'data/muv.csv')
data = list(dataset['Smiles'])

atom_num = atom_nums(data)
atom_dictionary = atom_dictionarys(atom_num)
cut_data=elect_data(data,15)

#数据导出
z= pd.DataFrame(cut_data)
z.to_csv("data/muv15.csv", index=None,sep=',')


dataset=pd.read_csv(r'data/muv15.csv')
dataset.columns=['Smiles']
data = list(dataset['Smiles'])

#转化分子为kekule式
from process_data import mol_datas,smile_datas

mol_data = mol_datas(data)
smile_data = smile_datas(mol_data)
    

dictionary = {}
def added_to_dictionary(smile):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True
            
for i in range(len(smile_data)):
    smile = smile_data[i]
    added_to_dictionary(smile)    

vocabs = [ele for ele in dictionary]
print(vocabs)
print(len(vocabs))


