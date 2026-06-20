"""
dataset.py

PyTorch Dataset that loads high-resolution satellite images and produces
(LR, HR) tensor pairs for super-resolution training, using on-the-fly
synthetic degradation (see degradation.py).

Why on-the-fly degradation (not precomputed):
    1. Saves disk space -- important on Colab/Kaggle's limited storage.
    2. Acts as free data augmentation -- each epoch sees different blur/
       noise/compression parameters for the same HR image, improving
       generalization.

Expected output:
    Each __getitem__ call returns a tuple (lr_tensor, hr_tensor):
        lr_tensor: torch.FloatTensor, shape (3, H/scale, W/scale), range [0, 1]
        hr_tensor: torch.FloatTensor, shape (3, H, W), range [0, 1]
"""

from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degradation import degrade_image

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR uint8 OpenCV image to a normalized RGB float tensor.

    OpenCV reads images in BGR order by default; we convert to RGB since
    that's the convention PyTorch/torchvision models expect.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    # HWC -> CHW
    tensor = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()
    return tensor


class SatelliteSRDataset(Dataset):
    """
    Dataset of (LR, HR) image pairs generated from a directory of HR
    satellite images.

    Args:
        image_dir: directory containing HR images (searched recursively).
        scale_factor: SR scale factor (2 or 4), forwarded to degradation pipeline.
        hr_crop_size: if set, HR images are center-cropped/resized to this
            square size before degradation, ensuring consistent batch shapes.
        degradation_cfg: dict of degradation hyperparameters
            (blur_kernel, blur_sigma_range, noise_std_range,
             jpeg_quality_range, apply_jpeg). Falls back to sane defaults
             if not provided.
    """

    def __init__(
        self,
        image_dir: str | Path,
        scale_factor: int = 4,
        hr_crop_size: int | None = 256,
        degradation_cfg: dict | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.scale_factor = scale_factor
        self.hr_crop_size = hr_crop_size
        self.degradation_cfg = degradation_cfg or {}

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory '{self.image_dir}' does not exist. "
                f"Check your config's data.raw_dir path."
            )

        self.image_paths = sorted(
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No supported images found under '{self.image_dir}'. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_hr_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(
                f"Failed to read image at '{path}'. File may be corrupted "
                f"or in an unsupported format."
            )

        if self.hr_crop_size is not None:
            h, w = image.shape[:2]
            # Resize first if image is smaller than target crop size,
            # then center-crop to guarantee exact, consistent dimensions.
            if h < self.hr_crop_size or w < self.hr_crop_size:
                scale = self.hr_crop_size / min(h, w)
                image = cv2.resize(
                    image, (int(w * scale) + 1, int(h * scale) + 1),
                    interpolation=cv2.INTER_CUBIC,
                )
                h, w = image.shape[:2]

            top = (h - self.hr_crop_size) // 2
            left = (w - self.hr_crop_size) // 2
            image = image[top: top + self.hr_crop_size, left: left + self.hr_crop_size]

        # Ensure HR dimensions are divisible by scale_factor to avoid
        # LR/HR size mismatches downstream.
        h, w = image.shape[:2]
        h_adj = h - (h % self.scale_factor)
        w_adj = w - (w % self.scale_factor)
        image = image[:h_adj, :w_adj]

        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        hr_image = self._load_hr_image(path)

        lr_image = degrade_image(
            hr_image,
            scale_factor=self.scale_factor,
            blur_kernel=self.degradation_cfg.get("gaussian_blur_kernel", 5),
            blur_sigma_range=tuple(self.degradation_cfg.get("gaussian_blur_sigma_range", (0.2, 2.0))),
            noise_std_range=tuple(self.degradation_cfg.get("noise_std_range", (0.0, 0.05))),
            jpeg_quality_range=tuple(self.degradation_cfg.get("jpeg_quality_range", (60, 100))),
            apply_jpeg=self.degradation_cfg.get("apply_jpeg_compression", True),
        )

        lr_tensor = _to_tensor(lr_image)
        hr_tensor = _to_tensor(hr_image)

        return lr_tensor, hr_tensor


if __name__ == "__main__":
    # Manual smoke test using a temporary directory of dummy images.
    # Run with: python -m src.data.dataset  (from project root)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_path = Path(tmp_dir) / "dummy.png"
        dummy_img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(dummy_path), dummy_img)

        dataset = SatelliteSRDataset(image_dir=tmp_dir, scale_factor=4, hr_crop_size=256)
        lr, hr = dataset[0]

        print("Dataset length:", len(dataset))
        print("LR tensor shape:", lr.shape, "range:", lr.min().item(), "-", lr.max().item())
        print("HR tensor shape:", hr.shape, "range:", hr.min().item(), "-", hr.max().item())

        assert lr.shape == (3, 64, 64), "Unexpected LR tensor shape."
        assert hr.shape == (3, 256, 256), "Unexpected HR tensor shape."
        print("Smoke test passed.")
