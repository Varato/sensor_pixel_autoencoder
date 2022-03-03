from typing import Tuple
from pathlib import Path

import torch

from .sensor_pixel_autoencoder import SensorPixelEncoder, SensorPixelDecoder
from .train import TrainCheckPoint


_checkpoint = TrainCheckPoint(
    encoder=Path("./model_data/encoder_epoch_10.pt"),
    decoder=Path("./model_data/decoder_epoch_10.pt"),
    loss_trace=Path("./model_data/loss_trace_epoch_10.npy"),
    epoch=10
)

def get_encoder(obs_shape: Tuple[int, int, int]) -> SensorPixelEncoder:
    encoder = SensorPixelEncoder(obs_shape, num_filters=32, num_layers=3, encoding_dim=128)
    encoder.load_state_dict(torch.load(_checkpoint.encoder))
    encoder.eval()
    return encoder


def get_decoder(obs_shape: Tuple[int, int, int], 
                encoder_config: SensorPixelEncoder.EncoderConfig) -> SensorPixelDecoder:
    decoder = SensorPixelDecoder(obs_shape, encoder_config)
    decoder.load_state_dict(torch.load(_checkpoint.decoder))
    decoder.eval()
    return decoder
