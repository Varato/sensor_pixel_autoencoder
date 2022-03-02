from pathlib import Path
import imageio

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from raw_dataset import RawImgDataset
from sensor_pixel_autoencoder import SensorPixelEncoder, SensorPixelDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(m: nn.Module, model_path: Path):
    m.load_state_dict(torch.load(model_path))

def mse_loss(recon: torch.Tensor, target: torch.Tensor):
    if target.max() > 1.0:
        print(target.max())
        assert(False)

    return F.mse_loss(recon, target)


def viz(recon: torch.Tensor, target: torch.Tensor, idx: int):
    mse = mse_loss(recon, target)
    cmp = torch.cat((target, recon), dim=-1).detach().cpu().numpy().squeeze()
    imageio.imwrite(f"pixel_autoencoder_test/result_{idx}.png", cmp)
    # plt.imshow(cmp.squeeze(), cmap='gray')
    # plt.title("mse = {:.3e}".format(mse))
    # plt.show()


def test(dataset: RawImgDataset, checkpoint: dict = None):
    obs_shape = dataset.img_shape
    encoder = SensorPixelEncoder(obs_shape, num_filters=32, num_layers=3, encoding_dim=128).to(DEVICE)
    decoder = SensorPixelDecoder(obs_shape, encoder_config=encoder.config).to(DEVICE)

    if checkpoint is not None:
        load_model(encoder, checkpoint["encoder"])
        load_model(decoder, checkpoint["decoder"])
    
    encoder.eval()
    decoder.eval()

    dl = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
    num_batches = len(dl)
    dl_it = iter(dl)
    for batch_id, batch_imgs in enumerate(dl_it):
        print(f"batch {batch_id + 1}/{num_batches}")
        batch_imgs_device = batch_imgs.to(DEVICE)
        encoding = encoder(batch_imgs_device)
        recon = decoder(encoding)
        viz(recon[0], batch_imgs_device[0], batch_id)

        

if __name__ == "__main__":
    dataset_path = Path(r'C:%USERPROFILE%\Workspace\Zurich-RAW-to-DSLR-Dataset\test\huawei_raw')

    checkpoint = {
        "encoder": Path("./model_data/encoder_epoch_8.pt"),
        "decoder": Path("./model_data/decoder_epoch_8.pt"),
        "loss_trace": Path("./model_data/loss_trace_epoch_8.npy"),
        "epoch": 8
    }

    ds = RawImgDataset(dataset_path)
    test(ds, checkpoint=checkpoint)
