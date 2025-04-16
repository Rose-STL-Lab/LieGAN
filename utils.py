import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from numpy import *

# coscorr = lambda x,y: np.trace(x.T@y)/np.norm(x)/np.norm(y)

cos_corr = lambda x,y: torch.trace(x.T @ y) / torch.norm(x) / torch.norm(y)

# scale the tensor to have dummy position equal to 1
def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor

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
L_e = np.zeros((6, 4, 4))
k = 3
for i in range(3):
    for j in range(i):
        L_e[k,i,j] = 1
        L_e[k,j,i] = -1
        k += 1
for i in range(3):
    L_e[i,i,3] = 1
L_lorentz = torch.tensor(L_lorentz, dtype=torch.float32)
def getLorentzLieAlgebra():
    return L_lorentz
def getEuclideanLieALgebra():
    return L_e

def randomSO13pTransform(x, var=1):
    L = getLorentzLieAlgebra().to(x.device)
    z = var * torch.randn(x.shape[0], 6).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', L, z))
    return torch.einsum('bij,bkj->bki', g_z, x)

def randomSO3Transform(x):
    L = getLorentzLieAlgebra()[3:, :, :].to(x.device)
    z = torch.randn(x.shape[0], 3).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', L, z))
    return torch.einsum('bij,bkj->bki', g_z, x)

def fourier_similarity(img1, img2, eps=1e-8):
    if img1.ndim == 4:
        img1 = img1.squeeze(1)
    if img2.ndim == 4:
        img2 = img2.squeeze(1)
    fft1 = torch.fft.fftshift(torch.fft.fft2(img1), dim=(-2, -1))
    fft2 = torch.fft.fftshift(torch.fft.fft2(img2), dim=(-2, -1))
    log_mag1 = torch.log1p(torch.abs(fft1) + eps)
    log_mag2 = torch.log1p(torch.abs(fft2) + eps)
    log_mag1_flat = log_mag1.view(log_mag1.shape[0], -1)
    log_mag2_flat = log_mag2.view(log_mag2.shape[0], -1)
    sim = F.cosine_similarity(log_mag1_flat, log_mag2_flat, dim=-1)  # shape: (B,)
    return sim.mean()