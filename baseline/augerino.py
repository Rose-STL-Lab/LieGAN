# Modified from https://github.com/g-benton/learning-invariances/blob/master/augerino/models/aug_modules.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class AugAveragedModel(nn.Module):
    def __init__(self, model, aug, ncopies=4):
        super().__init__()
        self.aug = aug  # this should be LieGenerator
        self.model = model
        self.ncopies = ncopies
    def forward(self, x, y):
        if self.training:
            gx, gy = self.aug(x, y)
            return gx, self.model(gx), gy
        else:
            raise NotImplementedError  # nothing to do with test time
            # bs = x.shape[0]
            # aug_x = torch.cat([self.aug(x) for _ in range(self.ncopies)],dim=0)
            # return sum(torch.split(F.log_softmax(self.model(aug_x),dim=-1),bs))/self.ncopies


class AugPredictionModel(nn.Module):
    def __init__(self, n_dim, n_input_timesteps, n_output_timesteps):
        super().__init__()
        self.n_dim = n_dim
        self.n_input_timesteps = n_input_timesteps
        self.n_output_timesteps = n_output_timesteps
        self.model = nn.Sequential(
            nn.Linear(n_dim * n_input_timesteps, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_dim * n_output_timesteps),
        )
    def forward(self, x):
        return self.model(x.view(-1, self.n_dim * self.n_input_timesteps)).view(-1, self.n_output_timesteps, self.n_dim)

            
class AugClassificationModel(nn.Module):
    def __init__(self, n_dim, n_components, n_classes):
        super().__init__()
        self.n_dim = n_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Linear(n_dim * n_components, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )
    def forward(self, x):
        return self.model(x.reshape(-1, self.n_dim * self.n_components))