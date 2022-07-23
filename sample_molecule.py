
import tensorflow as tf
import random
from convert_graph import graph_to_molecule

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole  


def sample(generator,batch_size,num_atoms,atom_dim,bond_dim,latent_dim):
    tf.random.set_seed(5)
    z = tf.random.normal((batch_size,latent_dim))
    graph = generator.predict(z)
    #获取独热编码
    adjacency = tf.argmax(graph[0], axis=1)
    adjacency = tf.one_hot(adjacency, depth=bond_dim, axis=1)
    adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    features = tf.argmax(graph[1], axis=2)
    features = tf.one_hot(features, depth=atom_dim, axis=2)
    return [graph_to_molecule([adjacency[i].numpy(), features[i].numpy()],
                              num_atoms,atom_dim,bond_dim) for i in range(batch_size)]

def draw_molecule(unique_molecules):
    mols=[]
    for smi in unique_molecules:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
    img = Draw.MolsToGridImage([m for m in mols if m is not None][:15], molsPerRow=5, subImgSize=(150, 150))
    return img

