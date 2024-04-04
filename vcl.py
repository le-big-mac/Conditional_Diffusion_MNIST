import os

import torch
from torch.utils import data

from mnist import get_split_MNIST
from model import ContextUnet, DDPM
from utils import eval, stack_params, train_epoch
import argparse

parser = argparse.ArgumentParser(description='Conditional Diffusion MNIST')
parser.add_argument('--save_dir', type=str, help='directory to save the results')
parser.add_argument('--mle_comp', action='store_true', help='whether to use compute MLE comparison')

args = parser.parse_args()

save_dir = args.save_dir
mle_comp = args.mle_comp
n_epoch = 20
batch_size = 256
n_T = 400 # 500
device = "cuda:0"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = False

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(n_classes):
    if not os.path.exists(f"{save_dir}/{i}"):
        os.makedirs(f"{save_dir}/{i}")

digit_datasets = get_split_MNIST()

if not mle_comp:
    # MLE pretraining
    nn_model = ContextUnet(1, n_feat, n_classes, mle=True)
    ddpm_mle = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm_mle.to(device)
    zero_loader = data.DataLoader(digit_datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm_mle.parameters(), lr=lrate)
    prior_mu, prior_logvar = stack_params(nn_model)
    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        train_epoch(ddpm_mle, zero_loader, optim, device)

    ddpm_mle.cpu()
else:
    prior_mu, prior_logvar = None, None

nn_model = ContextUnet(1, n_feat, n_classes, mle=mle_comp)
ddpm = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
if not mle_comp:
    ddpm.load_state_dict(ddpm_mle.state_dict())
ddpm.to(device)

for digit in range(n_classes):
    digit_data = digit_datasets[digit]
    digit_loader = data.DataLoader(digit_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        x, c = train_epoch(ddpm, digit_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle_comp)
        # save_gif = True if ep == n_epoch - 1 or ep%5 == 0 else False
        save_gif = False
        eval(ep, ddpm, digit+1, f"{save_dir}/{digit}/", device, save_gif=save_gif)

    prior_mu, prior_logvar = stack_params(nn_model)
