import torch
from torch import nn
# import emlp.nn.pytorch as enn
# from emlp.reps import Scalar, Vector
# from emlp.groups import Group
import numpy as np

class LGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout = 0., c_weight=1.0, last_layer=False, A=None, include_x=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 if not include_x else 10 # dims for Minkowski norm & inner product

        self.include_x = include_x
        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU())

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid())
        
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

        self.A = A
        self.norm_fn = normA_fn(A) if A is not None else normsq4
        self.dot_fn = dotA_fn(A) if A is not None else dotsq4

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        return out

    def m_model_extended(self, hi, hj, norms, dots, xi, xj):
        out = torch.cat([hi, hj, norms, dots, xi, xj], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = self.norm_fn(x_diff).unsqueeze(1)
        dots = self.dot_fn(x[i], x[j]).unsqueeze(1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)

        if self.include_x:
            m = self.m_model_extended(h[i], h[j], norms, dots, x[i], x[j])
        else:
            m = self.m_model(h[i], h[j], norms, dots) # [B*N, hidden]
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m

# # To replace LGEB with arbitrary group equivariant block
# class EMLPBlock(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden, G, n_node_attr=0,
#                  dropout = 0., c_weight=1.0, last_layer=False):
#         super(EMLPBlock, self).__init__()
#         self.c_weight = c_weight
#         n_edge_attr = 2 # dims for Minkowski norm & inner product
        
#         # replace phi_e net with EMLP
#         rep_in = (n_input * 2) * Scalar + 2 * Vector
#         rep_out = n_hidden * Scalar
#         self.phi_e = enn.EMLP(rep_in(G), rep_out(G), group=G, num_layers=2, ch=n_hidden)
#         # self.phi_e = nn.Sequential(
#         #     nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
#         #     nn.BatchNorm1d(n_hidden),
#         #     nn.ReLU(),
#         #     nn.Linear(n_hidden, n_hidden),
#         #     nn.ReLU())

#         self.phi_h = nn.Sequential(
#             nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
#             nn.BatchNorm1d(n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, n_output))

#         layer = nn.Linear(n_hidden, 1, bias=False)
#         torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

#         self.phi_x = nn.Sequential(
#             nn.Linear(n_hidden, n_hidden),
#             nn.ReLU(),
#             layer)

#         self.phi_m = nn.Sequential(
#             nn.Linear(n_hidden, 1),
#             nn.Sigmoid())
        
#         self.last_layer = last_layer
#         if last_layer:
#             del self.phi_x

#     def m_model(self, hi, hj, xi, xj):
#         out = torch.cat([hi, hj, xi, xj], dim=1)
#         out = self.phi_e(out)
#         w = self.phi_m(out)
#         out = out * w
#         return out

#     def h_model(self, h, edges, m, node_attr):
#         i, j = edges
#         agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
#         agg = torch.cat([h, agg, node_attr], dim=1)
#         out = h + self.phi_h(agg)
#         return out

#     def x_model(self, x, edges, x_diff, m):
#         i, j = edges
#         trans = x_diff * self.phi_x(m)
#         # From https://github.com/vgsatorras/egnn
#         # This is never activated but just in case it explosed it may save the train
#         trans = torch.clamp(trans, min=-100, max=100)
#         agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
#         x = x + agg * self.c_weight
#         return x

#     def x_diff_feat(self, edges, x):
#         i, j = edges
#         return x[i] - x[j]

#     def forward(self, h, x, edges, node_attr=None):
#         i, j = edges
#         x_diff = self.x_diff_feat(edges, x)

#         m = self.m_model(h[i], h[j], x[i], x[j]) # [B*N, hidden]
#         if not self.last_layer:
#             x = self.x_model(x, edges, x_diff, m)
#         h = self.h_model(h, edges, m, node_attr)
#         return h, x, m

# class CustomGroup(Group):
#     def __init__(self, generators):
#         if len(generators.shape) == 2:
#             generators = np.expand_dims(generators, axis=0)
#         self.lie_algebra = generators
#         n = generators.shape
#         super().__init__(n)

class LorentzNet(nn.Module):
    r''' Implementation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, n_scalar, n_hidden, n_class = 2, n_layers = 6, c_weight = 1e-3, dropout = 0., A=None, include_x=False):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([LGEB(self.n_hidden, self.n_hidden, self.n_hidden, 
                                    n_node_attr=n_scalar, dropout=dropout,
                                    c_weight=c_weight, last_layer=(i==n_layers-1), A=A, include_x=include_x)
                                    for i in range(n_layers)])
        self.graph_dec = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(self.n_hidden, n_class)) # classification

    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        h = self.embedding(scalars)

        for i in range(self.n_layers):
            # print(i)
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=scalars)

        h = h * node_mask
        h = h.view(-1, n_nodes, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)

def normA_fn(A):
    return lambda p: torch.einsum('...i, ij, ...j->...', p, A, p)

def dotA_fn(A):
    return lambda p, q: torch.einsum('...i, ij, ...j->...', p, A, q)
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)
