import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import math
from collections.abc import Iterable
from tqdm import tqdm, trange
from utils import *

def train_lie_gan(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    lr_d,
    lr_g,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    task='clf',
    save_path=None,
    print_every=100,
):
    # Loss function
    adversarial_loss = torch.nn.BCELoss(reduction='mean')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    train_generator = train_discriminator = True

    for ep in trange(num_epochs):
        D_real_loss_list, D_fake_loss_list, G_loss_list, G_reg_list, G_spreg_list, G_chreg_list = [], [], [], [], [], []
        for i, (x, y) in enumerate(dataloader):
            bs = x.shape[0]
            # Adversarial ground truths
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)
            # Configure input
            x = x.to(device)
            y = y.to(device)
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate a batch of transformed data points
            gx, gy = generator(x, y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gx, gy), valid)
            g_spreg = mu * torch.norm(generator.getLi(), p=p_norm)
            if reg_type == 'cosine':
                g_reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
            elif reg_type == 'rel_diff':
                g_reg = -torch.minimum(torch.abs((gx - x) / x).mean(), torch.FloatTensor([1.0]).to(device))
            elif reg_type == 'Li_norm':
                g_reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
            else:
                raise NotImplementedError
            g_reg = lamda * g_reg
            g_chreg = eta * generator.channel_corr(killing=False)
            G_loss_list.append(g_loss.item())
            G_reg_list.append(g_reg.item() / max(lamda, 1e-6))
            G_spreg_list.append(g_spreg.item() / max(mu, 1e-6))
            G_chreg_list.append(g_chreg.item() / max(eta, 1e-6))
            g_loss = g_loss + g_reg + g_spreg + g_chreg
            if train_generator:
                g_loss.backward()
                optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(x, y), valid)
            fake_loss = adversarial_loss(discriminator(gx.detach(), gy.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            if train_discriminator:
                d_loss.backward()
                optimizer_D.step()
            D_real_loss_list.append(real_loss.item())
            D_fake_loss_list.append(fake_loss.item())
        if save_path is not None and (ep + 1) % 100 == 0:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(generator.state_dict(), save_path + f'generator_{ep}.pt')
            torch.save(discriminator.state_dict(), save_path + f'discriminator_{ep}.pt')
        if (ep + 1) % print_every == 0:
            print(f'Epoch {ep}: D_real_loss={np.mean(D_real_loss_list)}, D_fake_loss={np.mean(D_fake_loss_list)}, G_loss={np.mean(G_loss_list)}, G_reg={np.mean(G_reg_list)}, G_spreg={np.mean(G_spreg_list)}, G_chreg={np.mean(G_chreg_list)}')
            print(generator.getLi())


def train_lie_gan_incremental(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    lr_d,
    lr_g,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    task='clf',
    save_path=None,
    print_every=100,
):
    # Loss function
    adversarial_loss = torch.nn.BCELoss(reduction='mean')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    train_generator = train_discriminator = True

    for ch in range(generator.n_channel):
        generator.set_activated_channel(ch)
        print(f'Training channel {ch}')
        for ep in trange(num_epochs):
            D_real_loss_list, D_fake_loss_list, G_loss_list, G_reg_list, G_spreg_list, G_chreg_list = [], [], [], [], [], []
            for i, (x, y) in enumerate(dataloader):
                # Adversarial ground truths
                valid = torch.ones(x.shape[0], 1, device=device)
                fake = torch.zeros(x.shape[0], 1, device=device)
                # Configure input
                x = x.to(device)
                y = y.to(device)
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Generate a batch of transformed data points
                gx, gy = generator(x, y)
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gx, gy), valid)
                g_spreg = mu * torch.norm(generator.getLi(), p=p_norm)
                if reg_type == 'cosine':
                    g_reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
                elif reg_type == 'rel_diff':
                    g_reg = -torch.minimum(torch.abs((gx - x) / x).mean(), torch.FloatTensor([1.0]).to(device))
                elif reg_type == 'Li_norm':
                    g_reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
                else:
                    raise NotImplementedError
                g_reg = lamda * g_reg
                g_chreg = eta * generator.channel_corr(killing=False)
                G_loss_list.append(g_loss.item())
                G_reg_list.append(g_reg.item() / max(lamda, 1e-6))
                G_spreg_list.append(g_spreg.item() / max(mu, 1e-6))
                G_chreg_list.append(g_chreg.item() / max(eta, 1e-6))
                g_loss = g_loss + g_reg + g_spreg + g_chreg
                if train_generator:
                    g_loss.backward()
                    grad_mask = torch.zeros_like(generator.Li.grad, device=generator.Li.device)
                    grad_mask[ch, :, :] = 1.0
                    generator.Li.grad *= grad_mask  # set other channels to zero
                    optimizer_G.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(x, y), valid)
                fake_loss = adversarial_loss(discriminator(gx.detach(), gy.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                if train_discriminator:
                    d_loss.backward()
                    optimizer_D.step()
                D_real_loss_list.append(real_loss.item())
                D_fake_loss_list.append(fake_loss.item())
            if save_path is not None and (ep + 1) % 100 == 0:
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(generator.state_dict(), save_path + f'generator_{ep}.pt')
                torch.save(discriminator.state_dict(), save_path + f'discriminator_{ep}.pt')
            if (ep + 1) % print_every == 0:
                print(f'Epoch {ch}-{ep}: D_real_loss={np.mean(D_real_loss_list)}, D_fake_loss={np.mean(D_fake_loss_list)}, G_loss={np.mean(G_loss_list)}, G_reg={np.mean(G_reg_list)}, G_spreg={np.mean(G_spreg_list)}, G_chreg={np.mean(G_chreg_list)}')
                print(generator.getLi())
        generator.activate_all_channels()


def train_liegerino(
    model,
    dataloader,
    num_epochs,
    lr,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    task='clf',
    save_path=None,
    print_every=100,
):
    if task == 'top_tagging':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif task == 'traj_pred':
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = model.aug
    for ep in trange(num_epochs):
        loss_list, reg_list, spreg_list, chreg_list = [], [], [], []
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            gx, fgx, gy = model(x, y)
            loss = criterion(fgx, gy)
            spreg = mu * torch.norm(generator.getLi(), p=p_norm)
            if reg_type == 'cosine':
                reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
            elif reg_type == 'rel_diff':
                reg = torch.abs((gx - x) / x).mean()
            elif reg_type == 'Li_norm':
                reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
            else:
                raise NotImplementedError
            reg = lamda * reg
            chreg = eta * generator.channel_corr(killing=False)
            loss_list.append(loss.item())
            reg_list.append(reg.item() / max(lamda, 1e-6))
            spreg_list.append(spreg.item() / max(mu, 1e-6))
            chreg_list.append(chreg.item() / max(eta, 1e-6))
            loss = loss + reg + spreg + chreg
            loss.backward()
            optimizer.step()
        if save_path is not None and (ep + 1) % 100 == 0:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), save_path + f'model_{ep}.pt')
        if (ep + 1) % print_every == 0:
            print(f'Epoch {ep}: loss={np.mean(loss_list)}, reg={np.mean(reg_list)}, spreg={np.mean(spreg_list)}')
            print(generator.getLi())
        