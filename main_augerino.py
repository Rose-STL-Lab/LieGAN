import torch
import torch.nn as nn
import numpy as np
import wandb
import argparse
from torch.utils.data import DataLoader
from dataset import NBodyDataset, TopTagging
from gan import LieGenerator
from baseline.augerino import *
from train import train_lie_gan, train_liegerino


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg_type', type=str, default='cosine')
    parser.add_argument('--lamda', type=float, default=1e-2)
    parser.add_argument('--p_norm', type=float, default=1)
    parser.add_argument('--droprate_init', type=float, default=0.8)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--activate_threshold', action='store_true')
    parser.add_argument('--D_loss_threshold', type=float, default=0.25)
    parser.add_argument('--coef_dist', type=str, default='normal')
    parser.add_argument('--g_init', type=str, default='random')
    parser.add_argument('--sigma_init', type=float, default=1)
    parser.add_argument('--uniform_max', type=float, default=1)
    parser.add_argument('--normalize_Li', action='store_true')
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--task', type=str, default='traj_pred')
    parser.add_argument('--dataset_name', type=str, default='2body')
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--input_timesteps', type=int, default=1)
    parser.add_argument('--output_timesteps', type=int, default=1)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--n_component', type=int, default=1)
    parser.add_argument('--x_type', type=str, default='vector')
    parser.add_argument('--y_type', type=str, default='vector')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    if args.task == 'traj_pred':
        dataset = NBodyDataset(
            input_timesteps=args.input_timesteps,
            output_timesteps=args.output_timesteps,
            save_path=f'./data/hnn/{args.dataset_name}-orbits-dataset.pkl',
        )
        n_dim = 8
        n_channel = 1
    elif args.task == 'top_tagging':
        dataset = TopTagging(n_component=args.n_component)
        n_dim = 4
        n_channel = args.n_channel
        n_component = args.n_component
        d_input_size = n_dim * n_component
        n_class = 2
        emb_size = 32
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize generator and discriminator
    generator = LieGenerator(n_dim, n_channel, args).to(args.device)
    if args.task == 'traj_pred':
        pred = AugPredictionModel(n_dim, args.input_timesteps, args.output_timesteps).to(args.device)
    else:
        pred = AugClassificationModel(n_dim, args.n_component, 2).to(args.device)
    model = AugAveragedModel(pred, generator).to(args.device)
    generator.mu.requires_grad = False
    generator.sigma.requires_grad = False

    # Train
    train_liegerino(
        model,
        dataloader,
        args.num_epochs,
        args.lr,
        args.reg_type,
        args.lamda,
        args.p_norm,
        args.mu,
        args.eta,
        args.device,
        task=args.task,
        save_path=f'{args.save_path}/{args.save_name}/',
    )
