from turtle import forward
from typing import Tuple
from collections import namedtuple
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SensorPixelEncoder(nn.Module):
    EncoderConfig = namedtuple("EncoderConfig", ["kernel_size", "stride", "padding", "num_layers", "conv_output_shape", "encoding_dim"])

    def __init__(self, obs_shape: Tuple[int, int, int], num_filters: int, num_layers: int, encoding_dim: int) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.encoding_dim = encoding_dim

        # use this configuration so that every layer reduces the image size by 2
        self._conv_kernel_size = 3
        self._conv_padding = self._conv_kernel_size // 2
        self._conv_stride = 2
    
        self.conv_lists = nn.ModuleList([
            nn.Conv2d(obs_shape[0], num_filters, self._conv_kernel_size, stride=self._conv_stride, padding=self._conv_padding)])
        for _ in range(num_layers - 1):
            self.conv_lists.append(
                nn.Conv2d(num_filters, num_filters, self._conv_kernel_size, stride=self._conv_stride, padding=self._conv_padding))

        conv_output_shape = self.conv_output_shape
        conv_output_flatten_dim = math.prod(conv_output_shape)
        self.fc = nn.Linear(conv_output_flatten_dim, encoding_dim)
        self.ln = nn.LayerNorm(encoding_dim)

    def init_parameters(self):
        self.apply(init_weights)

    @property
    def config(self):
        return SensorPixelEncoder.EncoderConfig(
            kernel_size=self._conv_kernel_size,
            stride=self._conv_stride,
            padding=self._conv_padding,
            num_layers=self.num_layers,
            conv_output_shape=self.conv_output_shape,
            encoding_dim = self.encoding_dim
        )

    @property
    def conv_output_shape(self):
        h, w = self.obs_shape[-2:]
        for _ in range(self.num_layers):
            h = (h + 2 * self._conv_padding - self._conv_kernel_size) // 2 + 1
            w = (w + 2 * self._conv_padding - self._conv_kernel_size) // 2 + 1
        return (self.num_filters, h, w)

    def _conv_fwd(self, x) -> torch.Tensor:
        # print("input:  ", x.shape)
        for i, conv in enumerate(self.conv_lists):
            x = F.relu(conv(x))
            # print(f"layer {i}:", x.shape)
        return x

    def forward(self, x)  -> torch.Tensor:
        x = self._conv_fwd(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)
        # print("hidden:", x.shape)
        return x


class SensorPixelDecoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], encoder_config: SensorPixelEncoder.EncoderConfig) -> None:
        super().__init__()

        self.encoder_config = encoder_config

        num_filters, h, w = self.encoder_config.conv_output_shape
        self.fc = nn.Linear(encoder_config.encoding_dim, num_filters * h * w)

        ksize = encoder_config.kernel_size
        stride = encoder_config.stride
        padding = encoder_config.padding

        self.t_conv_list = nn.ModuleList()
        for _ in range(encoder_config.num_layers - 1):
            self.t_conv_list.append(
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size=ksize, stride=stride, padding=padding, output_padding=1)
            )
        self.t_conv_list.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], kernel_size=ksize, stride=stride, padding=padding, output_padding=1)
        )

    def init_parameters(self):
        self.apply(init_weights)

    def _t_conv_fwd(self, x):
        # print("input:  ", x.shape)
        for i, t_conv in enumerate(self.t_conv_list[:-1]):
            x = F.relu(t_conv(x))
            # print(f"layer {i}:", x.shape)

        x = self.t_conv_list[-1](x)
        x = torch.sigmoid(x)
        # print(f"layer {-1}:", x.shape)
        return x

    def forward(self, x)  -> torch.Tensor:
        x = F.relu(self.fc(x))
        x = x.view(-1, *self.encoder_config.conv_output_shape)
        x = self._t_conv_fwd(x)
        return x




if __name__ == "__main__":
    h = 1440
    w = 3840
    batch = 10
    img = np.random.rand(h, w).astype(np.float32)
    img = utils.image_rescale(img)

    img = torch.from_numpy(img).to(DEVICE)
    img = img[None, None, ...]

    encoder = SensorPixelEncoder(obs_shape=img.shape[-3:], num_filters=32, num_layers=4, encoding_dim=128).to(DEVICE)
    decoder = SensorPixelDecoder(obs_shape=img.shape[-3:], encoder_config=encoder.config).to(DEVICE)
    h = encoder(img)
    img_d = decoder(h)

    # print(utils.transposed_conv_output_size(w=28, k=2, p=0, s=2))

