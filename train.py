import os
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


from raw_dataset import RawImgDataset
from sensor_pixel_autoencoder import SensorPixelEncoder, SensorPixelDecoder


TrainCheckPoint = namedtuple("TrainCheckPoint", ["encoder", "decoder", "loss_trace", "epoch"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_model(m: nn.Module, save_path: Path, name: str):
    if os.path.isdir(save_path):
        torch.save(m.state_dict(), save_path / name)
        print(f"modeld saved to {save_path / name}")
    else:
        os.mkdir(save_path)


def load_model(m: nn.Module, model_path: Path):
    if os.path.isfile(model_path):
        m.load_state_dict(torch.load(model_path))
    else:
        raise ValueError(f"{model_path} does not exist")


def mse_loss(recon: torch.Tensor, target: torch.Tensor):
    if target.max() > 1.0:
        raise ValueError("the images are not normalized to [0, 1]")
    return F.mse_loss(recon, target)


def train(dataset: RawImgDataset, num_epochs: int, lr: float = 1e-4, save_path = Path("."), checkpoint: TrainCheckPoint = None):
    obs_shape = dataset.img_shape
    encoder = SensorPixelEncoder(obs_shape, num_filters=32, num_layers=3, encoding_dim=128).to(DEVICE)
    decoder = SensorPixelDecoder(obs_shape, encoder_config=encoder.config).to(DEVICE)

    if checkpoint is not None:
        load_model(encoder, checkpoint.encoder)
        load_model(decoder, checkpoint.decoder)
        print("model loaded from {}".format(checkpoint.encoder))
        print("model loaded from {}".format(checkpoint.decoder))
        loss_trace = list(np.load(checkpoint.loss_trace))
        start_epoch = checkpoint.epoch
    else:
        decoder.init_parameters()
        encoder.init_parameters()
        loss_trace = []
        start_epoch = 0
    
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    dl = DataLoader(dataset, shuffle=True, batch_size=256, num_workers=1)
    num_batches = len(dl)
    for epoch in range(start_epoch, num_epochs):
        dl_it = iter(dl)
        for batch_id, batch_imgs in enumerate(dl_it):
            batch_imgs_device = batch_imgs.to(DEVICE)
            encoding = encoder(batch_imgs_device)
            recon = decoder(encoding)

            loss: torch.Tensor = mse_loss(recon, batch_imgs_device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            loss_trace.append(loss.detach().cpu().numpy())
            print(f"epoch {epoch + 1}/{num_epochs}, batch {batch_id + 1}/{num_batches}: loss = {loss.item():.3e}")
        save_model(encoder, save_path=save_path, name=f"encoder_epoch_{epoch+1}.pt")
        save_model(decoder, save_path=save_path, name=f"decoder_epoch_{epoch+1}.pt")
        np.save(save_path / f"loss_trace_epoch_{epoch+1}", np.float64(loss_trace))
        

if __name__ == "__main__":
    dataset_path = Path.home() / Path('/Workspace/Zurich-RAW-to-DSLR-Dataset/train/huawei_raw')
    ds = RawImgDataset(dataset_path)

    # checkpoint = TrainCheckPoint(
    #     encoder=Path("./model_data/encoder_epoch_1.pt"),
    #     decoder=Path("./model_data/decoder_epoch_1.pt"),
    #     loss_trace=Path("./model_data/loss_trace_epoch_1.npy"),
    #     epoch=1
    # )
    checkpoint = None

    train(ds, num_epochs=10, lr=1e-4, save_path=Path("model_data/"), checkpoint=checkpoint)

    

