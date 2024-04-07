from copy import deepcopy
import os
import pickle

import torch
from torch.utils import data

from mnist import get_split_MNIST, get_random_coreset
from utils import eval, model_setup, stack_params, train_epoch
import argparse

parser = argparse.ArgumentParser(description='Conditional Diffusion MNIST')
parser.add_argument('--save_dir', type=str, help='directory to save the results')
parser.add_argument('--mle_comp', action='store_true', help='whether to use compute MLE comparison')
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--gamma', type=float, default=1.0, help='batch size')
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_eval_samples', type=int, default=50, help='number of evaluation samples')
parser.add_argument('--deterministic_embed', action='store_true', help='whether to use deterministic embeddings')
parser.add_argument('--logvar_init', type=float, default=-10.0, help='initial logvar value')
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

hparams = {
    "n_feat": n_feat,
    "n_classes": n_classes,
    "deterministic_embed": deterministic_embed,
    "logvar_init": logvar_init
}

coreset_size = args.coreset_size
fashion = args.fashion

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.makedirs(save_dir, exist_ok=True)

for i in range(n_classes):
    os.makedirs(f"{save_dir}/{i}", exist_ok=True)
os.makedirs(f"{save_dir}/mle/", exist_ok=True)

digit_datasets = get_split_MNIST(fashion=fashion)
if coreset_size > 0:
    digit_datasets, coresets = zip(*[(get_random_coreset(digit_data, coreset_size)) for digit_data in digit_datasets])
    with open(f"{save_dir}/data_and_coresets.pkl", "wb+") as f:
        pickle.dump((digit_datasets, coresets), f)

if not mle_comp:
    # MLE pretraining for VCL
    ddpm = model_setup(hparams, mle=True, n_T=n_T, device=device)
    ddpm.to(device)
    zero_loader = data.DataLoader(digit_datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    for ep in range(20):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/20)
        train_epoch(ddpm, zero_loader, optim, device, num_param_samples=1)
    eval(19, ddpm, 1, f"{save_dir}/mle", device, num_eval_samples=1)
    prior_mu, prior_logvar = stack_params(ddpm)
    prior_mu, prior_logvar = prior_mu.detach().clone(), prior_logvar.detach().clone()
    ddpm.cpu()
    prev_params = deepcopy(ddpm.state_dict())
else:
    prior_mu, prior_logvar = None, None

if device == "cuda:0":
    torch.cuda.empty_cache()

ddpm = model_setup(hparams, mle=mle_comp, n_T=n_T, device=device)
if not mle_comp:
    ddpm.load_state_dict(prev_params)
ddpm.to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
prev_optim_state = deepcopy(optim.state_dict())

for digit in range(n_classes):
    digit_data = digit_datasets[digit]
    digit_loader = data.DataLoader(digit_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # Reinitialize with previous parameters if we are using coresets
    if coreset_size > 0 and digit > 0:
        ddpm.load_state_dict({k : v.to(device) for k, v in prev_params.items()})
        optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
        optim.load_state_dict(prev_optim_state)

    # Train on non-coreset data
    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        train_epoch(ddpm, digit_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle_comp, num_param_samples=num_param_samples, gamma=gamma)
        # save_gif = True if ep == n_epoch - 1 or ep%5 == 0 else False
        if ep % log_freq == 0 or ep == n_epoch - 1:
            save_gif = False
            eval(ep, ddpm, digit+1, f"{save_dir}/{digit}", device, save_gif=save_gif, num_eval_samples=num_eval_samples)

    # Extract prior for next class
    prior_mu, prior_logvar = stack_params(ddpm)
    prior_mu, prior_logvar = prior_mu.detach().clone(), prior_logvar.detach().clone()

    # Save model and prior
    if save_model:
        torch.save(ddpm.state_dict(), save_dir + f"/{digit}/model.pth")
        torch.save(optim.state_dict(), save_dir + f"/{digit}/optim.pth")
        with open(save_dir + f"/{digit}/prior.pkl", "wb+") as f:
            pickle.dump((prior_mu, prior_logvar), f)
        print('saved model at ' + save_dir + f"/model_{digit}.pth")

    # Train and evaluate on coreset data
    if coreset_size > 0:
        # Save pre-coreset parameters
        prev_params = deepcopy({k : v.cpu() for k, v in ddpm.state_dict().items()})
        prev_optim_state = deepcopy(optim.state_dict())
        for i in range(digit + 1):
            ddpm.load_state_dict({k : v.to(device) for k, v in prev_params.items()})
            optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
            optim.load_state_dict(prev_optim_state)
            coreset_loader = data.DataLoader(coresets[i], batch_size=batch_size, shuffle=True, num_workers=0)
            for ep in range(n_epoch):
                print(f"Epoch {ep}")
                optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
                train_epoch(ddpm, coreset_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle_comp, num_param_samples=num_param_samples, gamma=gamma)
            eval(ep, ddpm, digit+1, f"{save_dir}/{digit}", device, save_gif=save_gif, num_eval_samples=num_eval_samples, save_name=f"coreset_{i}")

    if device == "cuda:0":
        torch.cuda.empty_cache()
