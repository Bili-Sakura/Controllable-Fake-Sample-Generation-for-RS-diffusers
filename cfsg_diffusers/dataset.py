from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PairedImageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int, random_crop: bool) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _resize_or_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if self.random_crop and h > self.image_size and w > self.image_size:
            top = np.random.randint(0, h - self.image_size + 1)
            left = np.random.randint(0, w - self.image_size + 1)
            return img.crop((left, top, left + self.image_size, top + self.image_size))
        return img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hr_path, cond_path = self.pairs[index]
        hr = self._to_tensor(self._resize_or_crop(self._load_rgb(hr_path)))
        cond = self._to_tensor(self._resize_or_crop(self._load_rgb(cond_path)))
        return {
            "hr": hr,
            "condition": cond,
            "hr_path": str(hr_path),
            "condition_path": str(cond_path),
        }


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXT and p.is_file()])


def make_dataset_pairs(dataroot: str, datatype: str, data_len: int = -1) -> List[Tuple[Path, Path]]:
    root = Path(dataroot)
    if datatype in {"random", "change", "crop", "multiple", "noise", "large_scale"}:
        hr_dir = root / "hr_256"
        cond_dir = root / "sr_32_256"
    elif datatype in {"infer", "infer_to128", "infer_noise"}:
        hr_dir = root / "images"
        cond_dir = root / "labels"
    elif datatype == "test_infer":
        hr_dir = root / "hr_save"
        cond_dir = root / "sr_save"
    else:
        raise ValueError(f"Unsupported datatype for diffusers path: {datatype}")

    hr_paths = _list_images(hr_dir)
    cond_paths = _list_images(cond_dir)

    if len(hr_paths) != len(cond_paths):
        raise ValueError(
            f"Mismatched pair counts: {hr_dir} has {len(hr_paths)} files, {cond_dir} has {len(cond_paths)} files"
        )

    pairs = list(zip(hr_paths, cond_paths))
    if data_len > 0:
        pairs = pairs[: min(data_len, len(pairs))]
    return pairs
