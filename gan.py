import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class LieGenerator(nn.Module):
    def __init__(self, n_dim, n_channel, args):
        super(LieGenerator, self).__init__()
        self.n_dim = n_dim
        self.n_channel = n_channel
        self.args = args
        self.sigma = nn.Parameter(torch.eye(n_channel, n_channel) * args.sigma_init)
        self.mu = nn.Parameter(torch.zeros(n_channel))
        self.uniform_max = args.uniform_max
        self.dummy_pos = None
        self.l0reg = False
        self.activated_channel = None  # default to all channel
        if args.g_init == 'random':
            self.Li = nn.Parameter(torch.randn(n_channel, n_dim, n_dim))
            nn.init.kaiming_normal_(self.Li)
        elif args.g_init == 'zero':
            self.Li = nn.Parameter(torch.zeros(n_channel, n_dim, n_dim))
        elif args.g_init == '2*2_factorization':
            if args.task in ['traj_pred', ]:
                self.mask = torch.block_diag(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))
                self.Li = nn.Parameter(torch.randn(n_channel, n_dim, n_dim))
            elif args.task in ['traj_pred_3body', ]:
                self.mask = torch.block_diag(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))
                self.Li = nn.Parameter(torch.randn(n_channel, n_dim, n_dim))
            else:
                raise NotImplementedError
        elif args.g_init == '4*4_factorization':  # for traj_pred, assuming no interference between q and p
            if args.task in ['traj_pred', ]:
                self.mask = torch.block_diag(torch.ones(4, 4), torch.ones(4, 4))
                p = torch.eye(8)
                p[4:6,2:4] = p[2:4,4:6] = torch.eye(2)
                p[2:4,2:4] = p[4:6,4:6] = 0
                self.mask = p @ self.mask @ p
                self.Li = nn.Parameter(torch.randn(n_channel, n_dim, n_dim) * 0.01)
                # nn.init.kaiming_normal_(self.Li)
        elif args.g_init == 'affine':
            self.Li = nn.Parameter(torch.rand(n_channel, n_dim, n_dim))
            nn.init.kaiming_normal_(self.Li)
            self.mask = torch.ones(n_dim, n_dim)
            self.mask[2, :] = 0
        else:
            raise NotImplementedError

    def set_activated_channel(self, ch):
        self.activated_channel = ch

    def activate_all_channels(self):
        self.activated_channel = None

    def normalize_factor(self):
        trace = torch.einsum('kdf,kdf->k', self.Li, self.Li)
        factor = torch.sqrt(trace / self.Li.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self):
        return self.Li / (self.normalize_factor() + 1e-6)

    def channel_corr(self, killing=False):
        Li = self.normalize_L()
        if not killing:
            ch = self.activated_channel
            if ch is None:
                return torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', Li, Li), diagonal=1)))
            else:
                return torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', Li, Li), diagonal=1))[:(ch+1), :(ch+1)])
        else:
            trxy = torch.triu(torch.einsum('bij,cji->bc', Li, Li), diagonal=1)
            trx = torch.einsum('kdd->k', Li)
            trx_try = torch.triu(torch.einsum('b,c->bc', trx, trx), diagonal=1)
            return torch.sum(torch.abs(trxy - 1 / self.n_dim * trx_try))  # sum of tr(XY)-tr(X)tr(Y)/n, i.e. B(X,Y)/2n

    def forward(self, x, y):  # random transformation on x
        # x: (batch_size, n_components, n_dim); y: (batch_size, n_components_y, n_dim)
        if len(x.shape) == 2:
            x.unsqueeze_(1)
        if len(y.shape) == 2:
            y.unsqueeze_(1)
        batch_size = x.shape[0]
        z = self.sample_coefficient(batch_size, x.device)
        Li = self.normalize_L() if self.args.normalize_Li else self.Li
        if self.args.g_init in ['2*2_factorization', '4*4_factorization', 'affine', ]:
            g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, Li * self.mask.to(x.device)))
        else:
            g_z = torch.matrix_exp(torch.einsum('bj,jkl->bkl', z, Li))
        x_t = self.transform(g_z, x, self.args.x_type)
        y_t = self.transform(g_z, y, self.args.y_type)
        return x_t, y_t

    def sample_coefficient(self, batch_size, device):
        if self.args.coef_dist == 'normal':
            z = torch.randn(batch_size, self.n_channel, device=device) @ self.sigma + self.mu
        elif self.args.coef_dist == 'uniform':
            z = torch.rand(batch_size, self.n_channel, device=device) * 2 * self.uniform_max - self.uniform_max
        elif self.args.coef_dist == 'uniform_int_grid':
            z = torch.randint(-int(self.uniform_max), int(self.uniform_max), (batch_size, self.n_channel), device=device, dtype=torch.float32)
        ch = self.activated_channel
        if ch is not None:  # leaving only specified columns
            mask = torch.zeros_like(z, device=z.device)
            mask[:, ch] = 1
            z = z * mask
        return z
    
    def transform(self, g_z, x, tp):
        if tp == 'vector':
            return affine_coord(torch.einsum('bjk,btk->btj', g_z, x), self.dummy_pos)
        elif tp == 'scalar':
            return x
        elif tp == 'grid':
            grid = F.affine_grid(g_z[:, :-1], x.shape, align_corners=True)
            return F.grid_sample(x, grid, align_corners=True)

    def getLi(self):
        if self.args.g_init in ['2*2_factorization', '4*4_factorization', 'affine', ]:
            return self.Li * self.mask.to(self.Li.device)
        else:
            return self.Li


class LieDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(LieDiscriminator, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
        xy = torch.cat((x, y), dim=1)
        validity = self.model(xy)
        return validity


class LieDiscriminatorEmb(nn.Module):
    def __init__(self, input_size, n_class=2, emb_size=32):
        super(LieDiscriminatorEmb, self).__init__()
        self.input_size = input_size
        self.n_class = n_class
        self.emb_size = emb_size
        self.model = nn.Sequential(
            nn.Linear(input_size + emb_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.emb = nn.Embedding(n_class, emb_size)

    def forward(self, x, y):
        x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
        y = self.emb(y).squeeze(1)
        xy = torch.cat((x, y), dim=1)
        validity = self.model(xy)
        return validity
    

# Modified from Augerino codebase: https://github.com/g-benton/learning-invariances
class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def ConvBNrelu(chin, chout, stride=1):
    return nn.Sequential(
        nn.Conv2d(chin, chout, 3, padding=1, stride=stride),
        nn.BatchNorm2d(chout),
        nn.ReLU()
    )

class LieDiscriminatorConv(nn.Module):
    def __init__(self, chin=3, n_class=10, emb_size=128, k=128, dropout=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(chin,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            Expression(lambda u: u.mean(-1).mean(-1)),
        )
        self.emb = nn.Embedding(n_class, emb_size)
        self.lin = nn.Sequential(
            nn.Linear(2*k + emb_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        y = self.emb(y)
        x = self.net(x)
        xy = torch.cat((x, y), dim=1)
        return self.lin(xy)
