import os
import imageio
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

import utils


class RawImgDataset(Dataset):
    def __init__(self, dataset_path: Path, bit: int) -> None:
        super().__init__()

        self.path = Path(dataset_path)
        self.raw_files = [Path(x) for x in os.listdir(self.path) if x.endswith(".png")]
        self.raw_files.sort(key=lambda x: int(x.stem))

        tmp_img = self[0]
        self.img_shape = tmp_img.shape
        self.bit = bit
        self.pix_max = (1 << bit) - 1

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, index):
        bayer_raw = imageio.imread(self.path / self.raw_files[index])
        bayer_raw_norm = np.float32(bayer_raw) / self.pix_max
        r, gr, gb, b = utils.extract_bayer_channels(bayer_raw_norm)
        img = np.float32((gr + gb) * 0.5)[None, ...]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        return torch.from_numpy(img)
