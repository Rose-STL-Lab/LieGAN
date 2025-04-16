import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from utils import *
import pickle
import pandas as pd
import random


# Rotated MNIST
class RandRotShearTransform:
    def __init__(self, rot_range=(-90, 90), shear_range_x=(-30, 30), shear_range_y=(-30, 30)):
        self.rot_range = rot_range
        self.shear_range_x = shear_range_x
        self.shear_range_y = shear_range_y
        
    def __call__(self, x):
        rot_angle = random.uniform(*self.rot_range)
        shear_angle_x = random.uniform(*self.shear_range_x)
        shear_angle_y = random.uniform(*self.shear_range_y)
        # aff_generator = torch.FloatTensor([[0, -rot_angle + shear_angle_x, 0], [rot_angle + shear_angle_y, 0, 0], [0, 0, 0]])
        # aff_transform = torch.matrix_exp(aff_generator)[None, :-1]
        # grid = F.affine_grid(aff_transform, x[None].size())
        # return F.grid_sample(x[None], grid)[0]
        return TF.affine(x, angle=rot_angle, translate=(0, 0), scale=1, shear=(shear_angle_x, shear_angle_y))

def get_rotated_MNIST(rot_range=(-45, 45)):
    rt = transforms.Compose([
        transforms.ToTensor(),
        RandRotShearTransform(rot_range=rot_range, shear_range_x=(0, 0), shear_range_y=(0, 0)),
        # transforms.ToTensor()
    ])
    try:
        train_dataset = datasets.MNIST('./mnist', transform=rt, train=True)
        test_dataset = datasets.MNIST('./mnist', transform=rt, train=False)
    except RuntimeError:
        train_dataset = datasets.MNIST('./mnist', transform=rt, train=True, download=True)
        test_dataset = datasets.MNIST('./mnist', transform=rt, train=False, download=True)
    return train_dataset, test_dataset


# NBody
class NBodyDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='./data/hnn/2body-orbits-dataset.pkl', mode='train', trj_timesteps=50, input_timesteps=4, output_timesteps=1, extra_features=None, flatten=False, with_random_transform=False, nbody=2):
        with open(save_path, 'rb') as f:
            self.data = pickle.load(f)
        if mode == 'train':
            self.data = self.data['coords']
        else:
            self.data = self.data['test_coords']
        self.feat_dim = nbody * 4
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(-1, trj_timesteps, self.feat_dim)
        if nbody == 2:
            self.data = self.data[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
        elif nbody == 3:
            self.data = self.data[:, :, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.X, self.y = [], []
        self.N = self.data.shape[0]
        trj_timesteps = self.data.shape[1]
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps):
                self.X.append(self.data[i, t:t+input_timesteps, :])
                self.y.append(self.data[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        self.len = self.X.shape[0]
        if with_random_transform:
            if nbody == 2:
                GT = np.zeros((1, 8, 8))
                GT[0,1,0]=GT[0,3,2]=GT[0,5,4]=GT[0,7,6]=-1
                GT[0,0,1]=GT[0,2,3]=GT[0,4,5]=GT[0,6,7]=1
            elif nbody == 3:
                GT = np.zeros((1, 12, 12))
                GT[0,1,0]=GT[0,3,2]=GT[0,5,4]=GT[0,7,6]=GT[0,9,8]=GT[0,11,10]=-1
                GT[0,0,1]=GT[0,2,3]=GT[0,4,5]=GT[0,6,7]=GT[0,8,9]=GT[0,10,11]=1
            GT = torch.tensor(GT, dtype=torch.float32)
            z = torch.randn(self.X.shape[0], 1)
            g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', GT, z))
            self.gx = torch.einsum('bij,bkj->bki', g_z, self.X)
            self.gy = torch.einsum('bij,bkj->bki', g_z, self.y)
        if flatten:
            self.X = self.X.reshape(self.len, -1)
            self.y = self.y.reshape(self.len, -1)
            if with_random_transform:
                self.gx = self.gx.reshape(self.len, -1)
                self.gy = self.gy.reshape(self.len, -1)
        self.with_random_transform = with_random_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.with_random_transform:
            return self.X[idx], self.y[idx], self.gx[idx], self.gy[idx]
        else:
            return self.X[idx], self.y[idx]


# Top Tagging
class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=3, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = self.X * np.random.uniform(1-noise, 1+noise, size=self.X.shape)
        self.y = df[:, -1]
        self.X = torch.FloatTensor(self.X)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.y = torch.LongTensor(self.y)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Synthetic function invariant to discrete rotations
class DiscreteRotation(torch.utils.data.Dataset):
    def __init__(self, N=2000, k=7):
        self.X = np.random.randn(N, 3)
        self.y = np.arctan2(self.X[:, 1], self.X[:, 0])
        self.y = self.X[:, 2] / (1 + np.mod(self.y, 2 * np.pi / k))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
