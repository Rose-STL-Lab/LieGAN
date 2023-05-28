import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Generator(nn.Module):
    def __init__(self, n_dim, args):
        super(Generator, self).__init__()
        self.n_dim = n_dim
        self.args = args
        if args.g_init == 'random':
            self.Li = nn.Parameter(torch.randn(n_dim, n_dim))
        elif args.g_init == 'zero':
            self.Li = nn.Parameter(torch.zeros(n_dim, n_dim))
        elif args.g_init == 'identity':
            self.Li = nn.Parameter(torch.eye(n_dim))
        elif args.g_init == '2*2_factorization':
            if args.task == 'traj_pred':
                self.mask = torch.block_diag(torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))
                self.Li = nn.Parameter(torch.randn(n_dim, n_dim))
            else:
                raise NotImplementedError
        elif args.g_init == '4*4_factorization':  # for traj_pred, assuming no interference between q and p
            if args.task == 'traj_pred':
                self.mask = torch.block_diag(torch.ones(4, 4), torch.ones(4, 4))
                p = torch.eye(8)
                p[4:6,2:4] = p[2:4,4:6] = torch.eye(2)
                p[2:4,2:4] = p[4:6,4:6] = 0
                self.mask = p @ self.mask @ p
                self.Li = nn.Parameter(torch.zeros(n_dim, n_dim))

    def normalize_factor(self):
        trace = torch.einsum('df,df->', self.L, self.L)
        factor = torch.sqrt(trace / self.L.shape[1])
        return factor.unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self):
        return self.L / self.normalize_factor()

    def forward(self, x, y):  # random transformation on x
        # x: (batch_size, n_timesteps, n_dim); y: (batch_size, n_timesteps_y, n_dim)
        if len(x.shape) == 2:
            x.unsqueeze_(1)
        if len(y.shape) == 2:
            y.unsqueeze_(1)
        batch_size = x.shape[0]
        Li = self.normalize_L() if self.args.normalize_Li else self.Li
        if self.args.g_init in ['2*2_factorization', '4*4_factorization']:
            Li = Li * self.mask.to(x.device)
        return torch.einsum('jk,btk->btj', Li, x), torch.einsum('jk,btk->btj', Li, y)

    def getLi(self):
        if self.args.g_init in ['2*2_factorization', '4*4_factorization']:
            return self.Li * self.mask.to(self.Li.device)
        else:
            return self.Li