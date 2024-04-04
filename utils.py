import torch


def kl_divergence(mu1, logvar1, mu2, logvar2):
    log_std_diff = logvar2 - logvar1
    mu_diff_term =(torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2)
    return 0.5 * torch.sum(log_std_diff + mu_diff_term - 1)


def total_kl_divergence(new_model, old_model):
    kl = 0
    for new_layer, old_layer in zip(new_model.layers, old_model.layers):
        kl += kl_divergence(new_layer.weight_mean, new_layer.weight_logvar, old_layer.weight_mean, old_layer.weight_logvar)
        kl += kl_divergence(new_layer.bias_mean, new_layer.bias_logvar, old_layer.bias_mean, old_layer.bias_logvar)

    return kl