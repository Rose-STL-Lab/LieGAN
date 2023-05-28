import torch
import numpy as np

import h5py, glob

from torch.utils.data import ConcatDataset
from . import JetDataset

def initialize_datasets(datadir='./data', num_pts=None):
    """
    Initialize datasets.
    """

    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
    splits = ['train', 'test', 'valid'] # Our data categories -- training set, testing set and validation set
    patterns = {'train':'train', 'test':'test', 'valid':'val'} # Patterns to look for in data files, to identify which data category each belongs in
    
    files = glob.glob(datadir + '/*.h5')
    datafiles = {split:[] for split in splits}
    for file in files:
        for split,pattern in patterns.items():
            if pattern in file: datafiles[split].append(file)
    nfiles = {split:len(datafiles[split]) for split in splits}
    
    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!)
    #TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    if num_pts is None:
        num_pts={'train':-1,'test':-1,'valid':-1}
        
    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []
        
        if num_pts[split] == -1:
            for n in range(nfiles[split]): num_pts_per_file[split].append(-1)
        else:
            for n in range(nfiles[split]): num_pts_per_file[split].append(int(np.ceil(num_pts[split]/nfiles[split])))
            num_pts_per_file[split][-1] = int(np.maximum(num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]),0))
    
    ### ------ 3: Load the data ------ ###
    datasets = {}
    for split in splits:
        datasets[split] = []
        for file in datafiles[split]:
            with h5py.File(file,'r') as f:
                datasets[split].append({key: torch.from_numpy(val[:]) for key, val in f.items()})
 
    ### ------ 4: Error checking ------ ###
    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = []
    for split in splits:
        for dataset in datasets[split]:
            keys.append(dataset.keys())
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    ### ------ 5: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    torch_datasets = {split: ConcatDataset([JetDataset(data, num_pts=num_pts_per_file[split][idx]) for idx, data in enumerate(datasets[split])]) for split in splits}

    return torch_datasets

# Lorentz group Lie algebra
L_lorentz = np.zeros((6,4,4))
k = 3
for i in range(3):
    for j in range(i):
        L_lorentz[k,i+1,j+1] = 1
        L_lorentz[k,j+1,i+1] = -1
        k += 1
for i in range(3):
    L_lorentz[i,1+i,0] = 1
    L_lorentz[i,0,1+i] = 1
L_lorentz = torch.tensor(L_lorentz, dtype=torch.float32)
def getLorentzLieAlgebra():
    return L_lorentz

def randomSO13pTransform(x, var=1.0):
    L = getLorentzLieAlgebra().to(x.device)
    z = var * torch.randn(x.shape[0], 6).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', L, z)).type(x.dtype)
    return torch.einsum('bij,bkj->bki', g_z, x)

def randomLieGroupTransform(G, x, var=1.0):
    z = var * torch.randn(x.shape[0], G.shape[0]).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', G, z)).type(x.dtype)
    return torch.einsum('bij,bkj->bki', g_z, x)