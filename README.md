All the steps required to run Mask RCNN with GPU. 
Moreover, this read me file provide step by step procedure, how to install cuda and cudnn on Windows. 


#### Tensorflow gpu istallation
[Refernce Video Link ] [https://www.youtube.com/watch?v=hHWkvEcDBO0&ab_channel=AladdinPersson]


1. Microsoft Visual Studio
* https://visualstudio.microsoft.com/vs...

2. the NVIDIA CUDA Toolkit
* https://developer.nvidia.com/cuda-too...

3. NVIDIA cuDNN
* https://developer.nvidia.com/cudnn

4. Python (check compatible version from first link)
conda create --name tf_2.4 python==3.8


## Install conda and set up a TensorFlow 1.15, CUDA 10.0 environment on Ubuntu/Windows
[ https://fmorenovr.medium.com/install-conda-and-set-up-a-tensorflow-1-15-cuda-10-0-environment-on-ubuntu-windows-2a18097e6a98]
#### Following are the libraries needed to be installed in order to make Mask RCNN to work like a charm
* $ conda create --name tf1 python=3.7
* $ conda activate tf1
* $ conda install -c conda-forge tensorflow-gpu=1.15
* $ conda install -c conda-forge tensorboardx -y
* $ conda install -c conda-forge notebook -y
* $ conda install -c conda-forge numpy=1.16.6 -y
* $ conda install -c conda-forge pandas -y
* $ conda install -c conda-forge matplotlib -y
* $ conda install -c conda-forge opencv -y
* $ conda install -c conda-forge scikit-learn -y
* $ conda install -c conda-forge tqdm -y
* $ conda install -c conda-forge scikit-image -y
* $ conda install -c anaconda scipy=1.5.3 -y
* $ conda install -c anaconda h5py=2.10.0 -y
* $ pip install --ignore-installed jupyter
* $ pip install jupyter

# Some specific package 
* $conda install -c anaconda pywget -y
* $conda install -c conda-forge shapely

https://fmorenovr.medium.com/install-conda-and-set-up-a-tensorflow-1-15-cuda-10-0-environment-on-ubuntu-windows-2a18097e6a98
##
# How to Know if Keras is using GPU or CPU/ Steps for running GPU on Windows

#### First, you need to find the whther your system has a GPU or not:

physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')

#### then you can check if your GPU device is on Used for training or not with this code:

tf.config.experimental.get_memory_growth(physical_device[0])
tf.config.experimental.get_memory_growth(physical_device[1]) #in case you have two GPU cards, if you only have one GPU card then skip this line
#### if this code returns False or nothing then you can run this code below to set GPU for training

tf.config.experimental.set_memory_growth(physical_device[0],True)
tf.config.experimental.set_memory_growth(physical_device[1],True)


# Installing Pytorch with CUDA
link [https://varhowto.com/install-pytorch-cuda-10-0/]
pip install torch==1.2.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

#### MUST RESTART THE SYSTEM!