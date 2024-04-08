import argparse
from copy import deepcopy
import os
import pickle

import torch
from torch.utils import data

from mnist import get_split_MNIST, get_random_coreset, merge_datasets
from utils import eval, model_setup, stack_params, train_epoch

parser = argparse.ArgumentParser(description='Conditional Diffusion MNIST')
parser.add_argument('--save_dir', type=str, help='directory to save the results')
parser.add_argument('--mle_comp', action='store_true', help='whether to use compute MLE comparison')
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--gamma', type=float, default=1.0, help='batch size')
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_eval_samples', type=int, default=10, help='number of evaluation samples')
parser.add_argument('--deterministic_embed', action='store_true', help='whether to use deterministic embeddings')
parser.add_argument('--logvar_init', type=float, default=-12.0, help='initial logvar value')
parser.add_argument('--log_freq', type=int, default=20, help='logging frequency')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--coreset_size', type=int, default=0, help='size of coreset')
parser.add_argument('--fashion', action='store_true', help='use FashionMNIST instead of MNIST' )

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
log_freq = args.log_freq
batch_size = args.batch_size
n_T = 400 # 500
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
save_model = True
num_param_samples = 1 if mle_comp else 10

coreset_size = args.coreset_size
fashion = args.fashion

if coreset_size > 0:
    with open(f"{save_dir}/data_and_coresets.pkl", "rb") as f:
        digit_datasets, coresets = pickle.load(f)

hparams = {
    "n_feat": n_feat,
    "n_classes": n_classes,
    "deterministic_embed": deterministic_embed,
    "logvar_init": logvar_init
}

# No need to reinitialize model each class if we are not using coresets
ddpm = model_setup(hparams, mle=mle_comp, n_T=n_T, device=device)
ddpm.to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

for digit in range(n_classes):
    # Reinitialize with previous parameters if we are using coresets
    prev_params = torch.load(f"{save_dir}/{digit}/model.pth")
    prev_optim_state = torch.load(f"{save_dir}/{digit}/optim.pth")
    with open(f"{save_dir}/{digit}/prior.pkl", "rb") as f:
      prior_mu, prior_logvar = pickle.load(f)

    merged = merge_datasets(coresets[:(digit+1)])
    ddpm.load_state_dict({k : v.to(device) for k, v in prev_params.items()})
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    optim.load_state_dict(prev_optim_state)
    coreset_loader = data.DataLoader(merged, batch_size=batch_size, shuffle=True, num_workers=0)
    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        train_epoch(ddpm, coreset_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=True, num_param_samples=1, gamma=gamma)
    eval(ep, ddpm, digit+1, f"{save_dir}/{digit}", device, save_gif=False, num_eval_samples=num_eval_samples, save_name="coreset_mle_all", ws_test=[2.0])

    if device == "cuda:0":
        torch.cuda.empty_cache()