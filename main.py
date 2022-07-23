
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import random
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

from process_data import delet_element,elect_data
from process_data import data_to_mol,smiles_to_mol
from process_data import kekule_mol

from rdkit import Chem
from rdkit import RDLogger       
from rdkit.Chem.Draw import IPythonConsole  

from convert_graph import smiles_to_graph,graph_to_molecule
from convert_graph import convert_tensor

from sample_molecule import sample,draw_molecule
from calculate_property import check_valid,check_novelty,check_unique


dataset=pd.read_csv(r'data\QM9.csv')
data = list(dataset['smiles'])

# Delete molecules containing redundant elements
data ,delet_smile = delet_element(data)

# remove the value of None when converted to mol
# data_mol ,data_none= data_to_mol(train_data)

train_data = data

random.seed(2)
random.shuffle(train_data)

num_atoms = 9
atom_dim  = 5
bond_dim  = 5
latent_dim= 32

#example
smiles = train_data[95550]
print("SMILES:", smiles)
molecule = smiles_to_mol(smiles)
print("Num heavy atoms:", molecule.GetNumHeavyAtoms())
molecule

graph_to_molecule(smiles_to_graph(smiles,num_atoms,atom_dim,bond_dim),num_atoms,atom_dim,bond_dim)
molecule = Chem.MolFromSmiles(smiles)

adjacency_tensor,feature_tensor = convert_tensor(train_data,num_atoms,atom_dim,bond_dim)
print("adjacency_tensor.shape =", adjacency_tensor.shape)
print("feature_tensor.shape =", feature_tensor.shape)


from DRAGAN import GraphGenerator,GraphDiscriminator
from DRAGAN import GraphDRAGAN


generator = GraphGenerator(latent_dim=latent_dim,adjacency_shape=(bond_dim, num_atoms, num_atoms),
                           feature_shape=(num_atoms, atom_dim))

discriminator = GraphDiscriminator(adjacency_shape=(bond_dim, num_atoms, num_atoms),
                                   feature_shape=(num_atoms, atom_dim))

generator.summary()
discriminator.summary()

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True
        self.model = GraphDRAGAN(generator,discriminator)
 
    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "G_learning_rate" in logs or "D_learning_rate" in logs:
            return
        logs["G_learning_rate"] = self.model.optimizer_generator.lr
        logs["D_learning_rate"] = self.model.optimizer_discriminator.lr


wgan = GraphDRAGAN(generator, discriminator,discriminator_steps=1,generator_steps=1)
wgan.compile(optimizer_generator=keras.optimizers.Adam(lr=0.001,beta_1=0.5,beta_2=0.999),
              optimizer_discriminator=keras.optimizers.Adam(lr=0.0001,beta_1=0.5,beta_2=0.999))

model = wgan

# def scheduler(epoch):
#     # With every 100 epochs, the learning rate decreases by 1/10
#     if epoch % 100 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer_discriminator.lr)
#         K.set_value(model.optimizer_discriminator.lr, lr * 0.7)
#         print("discriminator_lr changed to {}".format(lr * 0.7))
#     return K.get_value(model.optimizer_discriminator.lr)

#model callbacks
log_file_path = './logs.csv'
tensorboard=tf.keras.callbacks.TensorBoard(histogram_freq=1,write_images=True)
csv_logger = CSVLogger(log_file_path,append=True)            #每个epoch的训练结果保存为csv

#reduce_lr = LearningRateScheduler(scheduler)
callbacks = [tensorboard,csv_logger,LearningRateLogger()]

wgan.fit([adjacency_tensor,feature_tensor], epochs=100, batch_size=32,callbacks=callbacks)


# cd C:\Users\19368\Desktop\graph_GAN
# tensorboard --logdir=logs/train
# http://localhost:6006/

wgan.save_weights('QM9')
new_model = wgan
new_model.load_weights('QM9')
new_model.fit([adjacency_tensor,feature_tensor], epochs=100, batch_size=32,callbacks=callbacks)
new_model.save_weights('QM9')


if __name__ == "__main__" :
    generator_molecules = sample(wgan.generator,batch_size=10000,num_atoms=num_atoms,
                                  atom_dim=atom_dim ,bond_dim=bond_dim,latent_dim=latent_dim)
    print ("{} generator molecules".format(len(generator_molecules)))
    
    valid_molecules = check_valid(generator_molecules)
    print ("{} valid molecules".format(len(valid_molecules)))
    
    novel_molecules = check_novelty(generator_molecules,data)
    print ("{} novel molecules".format(len(novel_molecules)))
    
    unique_molecules = check_unique(generator_molecules)
    print ("{} unique molecules".format(len(unique_molecules)))
    
img=draw_molecule(unique_molecules)
img    


