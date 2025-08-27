# STAGAN
STAGAN uses generative adversarial networks to perform small molecule map generation tasks, which are implemented in the Keras framework. It allows the user to run the model to generate a reference set of drug-like molecules.

# Requirement
```
Refer to requirement.txt
```

# Installation
* Install [python 3.7](https://www.python.org/downloads/) in Linux and Windows.
* If you want to run on a GPU, you will need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), please refer to their websites for corresponding versions.
* Add your installation path and run the following command to install the DivGAN libraries in one step
```
pip install -r requirement.txt
```

# Running STAGAN
You need to open main.py, run load_weights to read the pre-trained weights and get the generated molecules.
Or provide training set molecules into graph coding for model training.
