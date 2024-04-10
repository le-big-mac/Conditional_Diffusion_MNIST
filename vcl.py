from copy import deepcopy
import os
import pickle

import torch
from torch.utils import data

from data.mnist import get_split_MNIST, get_random_coreset, merge_datasets
from utils import eval, model_setup, sample_dataset, stack_params, train_epoch
import argparse

parser = argparse.ArgumentParser(description='Conditional Diffusion MNIST')
parser.add_argument('--save_dir', type=str, help='directory to save the results')
parser.add_argument('--mle', action='store_true', help='whether to use compute MLE comparison')
parser.add_argument('--coreset_size', type=int, default=0, help='size of coreset')
parser.add_argument('--fashion', action='store_true', help='use FashionMNIST instead of MNIST' )
parser.add_argument('--sample_datasets', action='store_true', help='sample datasets')
parser.add_argument('--save_model', action='store_true', help='save model')

args = parser.parse_args()
print(args)

save_dir = args.save_dir
mle = args.mle
coreset_size = args.coreset_size
fashion = args.fashion
sample_dset = args.sample_datasets
save_model = args.save_model
n_train_epoch = 25
n_coreset_epoch = 10
lrate = 1e-4
logvar_init = -10.0
batch_size = 128
n_T = 400
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_classes = 10
n_feat = 128
num_param_samples = 1 if mle else 10
num_eval_samples = 1 if mle else 10

hparams = {
    "n_feat": n_feat,
    "n_classes": n_classes,
    "logvar_init": logvar_init
}

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.makedirs(save_dir, exist_ok=True)

for i in range(n_classes):
    os.makedirs(f"{save_dir}/{i}", exist_ok=True)
os.makedirs(f"{save_dir}/mle/", exist_ok=True)
os.makedev(f"{save_dir}/sampled_datasets", exist_ok=True)


digit_datasets = get_split_MNIST(fashion=fashion)
if coreset_size > 0:
    digit_datasets, coresets = zip(*[(get_random_coreset(digit_data, coreset_size)) for digit_data in digit_datasets])
    with open(f"{save_dir}/data_and_coresets.pkl", "wb+") as f:
        pickle.dump((digit_datasets, coresets), f)

if not mle:
    # MLE pretraining for VCL
    ddpm = model_setup(hparams, mle=True, n_T=n_T, device=device)
    ddpm.to(device)
    zero_loader = data.DataLoader(digit_datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    for ep in range(20):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/20)
        train_epoch(ddpm, zero_loader, optim, device, num_param_samples=1)
    eval(ddpm, 1, f"{save_dir}/mle/training_end", device, num_param_samples=1)
    prior_mu, prior_logvar = stack_params(ddpm)
    prior_mu, prior_logvar = prior_mu.detach().clone(), prior_logvar.detach().clone()
    ddpm.cpu()
    prev_params = deepcopy(ddpm.state_dict())
else:
    prior_mu, prior_logvar = None, None

if device == "cuda:0":
    torch.cuda.empty_cache()

ddpm = model_setup(hparams, mle=mle, n_T=n_T, device=device)
if not mle:
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
    for ep in range(n_train_epoch):
        print(f"Epoch {ep}")
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_train_epoch)
        train_epoch(ddpm, digit_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle, num_param_samples=num_param_samples)
    eval(ddpm, digit+1, f"{save_dir}/{digit}/training_end", device, num_param_samples=num_eval_samples)

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
        optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
        merged = merge_datasets(coresets[:digit+1])
        coreset_loader = data.DataLoader(merged, batch_size=batch_size, shuffle=True, num_workers=0)
        for ep in range(n_coreset_epoch):
            print(f"Epoch {ep}")
            optim.param_groups[0]['lr'] = lrate*(1-ep/n_coreset_epoch)
            train_epoch(ddpm, coreset_loader, optim, device, prior_mu=prior_mu, prior_logvar=prior_logvar, mle=mle, num_param_samples=num_param_samples)
        eval(ddpm, digit+1, f"{save_dir}/{digit}/coreset_eval", device, num_param_samples=num_eval_samples)
        if sample_dset:
            sample_dataset(ddpm, digit+1, f"{save_dir}/sampled_datasets", device, 2.0, num_datapoints=500, num_param_samples=num_eval_samples)

    if device == "cuda:0":
        torch.cuda.empty_cache()
