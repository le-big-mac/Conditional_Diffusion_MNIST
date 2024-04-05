import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mle=False):
        super().__init__()
        self.mle = mle
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        nn.init.constant_(self.weight_logvar, -6.)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_logvar, -6.)

    def forward(self, x, num_param_samples=10):
        if self.mle:
            return F.linear(x, self.weight_mean, self.bias_mean)

        x = x.reshape(num_param_samples, -1, *x.shape[1:])

        def param_sample(x):
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(weight_std)
            bias = None
            if self.bias_mean is not None:
                bias_std = torch.exp(0.5 * self.bias_logvar)
                bias = self.bias_mean + bias_std * torch.randn_like(bias_std)

            return F.linear(x, weight, bias)

        return torch.vmap(param_sample, in_dims=0, randomness='different')(x).reshape(-1, *x.shape[2:])



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, mle=False):
        super().__init__()
        self.mle = mle
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_mean = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        nn.init.constant_(self.weight_logvar, -6.)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_logvar, -6.)

    def forward(self, x, num_param_samples=10):
        if self.mle:
            return F.conv2d(x, self.weight_mean, self.bias_mean, stride=self.stride, padding=self.padding)

        x = x.reshape(num_param_samples, -1, *x.shape[1:])

        def param_sample(x):
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(weight_std)
            bias = None
            if self.bias_mean is not None:
                bias_std = torch.exp(0.5 * self.bias_logvar)
                bias = self.bias_mean + bias_std * torch.randn_like(bias_std)
            return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

        return torch.vmap(param_sample, in_dims=0, randomness='different')(x).reshape(-1, *x.shape[2:])


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, mle=False):
        super().__init__()
        self.mle = mle
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_mean = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))
        self.weight_logvar = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        nn.init.constant_(self.weight_logvar, -6.)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_logvar, -6.)

    def forward(self, x, num_param_samples=10):
        if self.mle:
            return F.conv_transpose2d(x, self.weight_mean, self.bias_mean, stride=self.stride, padding=self.padding)

        x = x.reshape(num_param_samples, -1, *x.shape[1:])

        def param_sample(x):
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(weight_std)
            bias = None
            if self.bias_mean is not None:
                bias_std = torch.exp(0.5 * self.bias_logvar)
                bias = self.bias_mean + bias_std * torch.randn_like(bias_std)
            return F.conv_transpose2d(x, weight, bias, stride=self.stride, padding=self.padding)

        return torch.vmap(param_sample, in_dims=0, randomness='different')(x).reshape(-1, *x.shape[2:])
