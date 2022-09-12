import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, inputs):
        return fused_leaky_relu(inputs, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(inputs, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (inputs.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                inputs + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

    else:
        return F.leaky_relu(inputs, negative_slope=negative_slope) * scale