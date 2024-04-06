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
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--gamma', type=float, default=1.0, help='batch size')
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_eval_samples', type=int, default=50, help='number of evaluation samples')
parser.add_argument('--deterministic_embed', action='store_true', help='whether to use deterministic embeddings')
parser.add_argument('--logvar_init', type=float, default=-8.0, help='initial logvar value')

args = parser.parse_args()
print(args)

save_dir = args.save_dir
mle_comp = args.mle_comp
n_epoch = args.n_epoch
gamma = args.gamma
lrate = args.lrate
num_eval_samples = args.num_eval_samples
deterministic_embed = args.deterministic_embed
logvar_init = args.logvar_init
batch_size = 256
n_T = 400 # 500
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
save_model = True
num_param_samples = 1 if mle_comp else 10

os.makedirs(save_dir, exist_ok=True)

for i in range(n_classes):
    os.makedirs(f"{save_dir}/{i}", exist_ok=True)
os.makedirs(f"{save_dir}/mle/", exist_ok=True)

digit_datasets = get_split_MNIST()

if not mle_comp:
    # MLE pretraining
    nn_model = ContextUnet(1, n_feat, n_classes, mle=True, deterministic_embed=deterministic_embed, logvar_init=logvar_init)
    ddpm_mle = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm_mle.to(device)
    zero_loader = data.DataLoader(digit_datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm_mle.parameters(), lr=lrate)
    for ep in range(20):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        train_epoch(ddpm_mle, zero_loader, optim, device, num_param_samples=1)
    eval(19, ddpm_mle, 1, f"{save_dir}/mle/", device, num_eval_samples=1)
    prior_mu, prior_logvar = stack_params(nn_model)
    prior_mu, prior_logvar = prior_mu.detach().clone(), prior_logvar.detach().clone()
    ddpm_mle.cpu()
else:
    prior_mu, prior_logvar = None, None

nn_model = ContextUnet(1, n_feat, n_classes, mle=mle_comp, deterministic_embed=deterministic_embed, logvar_init=logvar_init)
ddpm = DDPM(nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
if not mle_comp:
    ddpm.load_state_dict(ddpm_mle.state_dict())
ddpm.to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

for digit in range(n_classes):
    digit_data = digit_datasets[digit]
    digit_loader = data.DataLoader(digit_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        x, c = train_epoch(ddpm, digit_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle_comp, num_param_samples=num_param_samples, gamma=gamma)
        # save_gif = True if ep == n_epoch - 1 or ep%5 == 0 else False
        if ep % 20 == 0 or ep == n_epoch - 1:
            save_gif = False
            eval(ep, ddpm, digit+1, f"{save_dir}/{digit}/", device, save_gif=save_gif, num_eval_samples=num_eval_samples)

    if save_model:
        torch.save(ddpm.cpu().state_dict(), save_dir + f"model_{digit}.pth")
        print('saved model at ' + save_dir + f"model_{digit}.pth")

    prior_mu, prior_logvar = stack_params(nn_model)
    prior_mu, prior_logvar = prior_mu.detach().clone(), prior_logvar.detach().clone()
