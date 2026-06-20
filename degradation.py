"""
degradation.py

Synthetically generates a low-resolution (LR) image from a high-resolution
(HR) image, simulating real-world satellite image degradation.

Why synthetic degradation:
    Public satellite datasets (UC Merced, NWPU-RESISC45, etc.) only provide
    high-resolution images -- there is no naturally paired low-res version.
    The standard approach in super-resolution research (used by DIV2K, the
    most common SR benchmark dataset) is to synthetically degrade HR images
    to create training pairs. We mimic plausible real degradation:

        1. Gaussian blur   -> simulates atmospheric / optical blur
        2. Bicubic downsample -> simulates lower sensor resolution
        3. Gaussian noise  -> simulates sensor noise
        4. JPEG compression -> simulates lossy transmission/storage artifacts

    Randomizing parameters per-sample (rather than using fixed values) acts
    as a form of data augmentation, improving generalization.

Expected output:
    Given an HR image of shape (H, W, 3) in [0, 255] uint8, `degrade_image`
    returns an LR image of shape (H/scale, W/scale, 3) in [0, 255] uint8.
"""

from __future__ import annotations
import cv2
import numpy as np
import random


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply Gaussian blur. kernel_size must be odd."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # cv2 requires odd kernel sizes
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def downsample(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Downsample image by scale_factor using bicubic interpolation."""
    h, w = image.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    if new_h < 1 or new_w < 1:
        raise ValueError(
            f"Image too small ({h}x{w}) to downsample by factor {scale_factor}."
        )
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def add_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """
    Add Gaussian sensor noise. Operates on float [0,1] scale internally
    to avoid uint8 wraparound artifacts, then converts back.
    """
    if std <= 0:
        return image
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=img_float.shape).astype(np.float32)
    noisy = np.clip(img_float + noise, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Simulate JPEG compression artifacts by encoding and immediately
    decoding the image in memory (no file I/O).
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_img = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed during degradation simulation.")
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return decoded_img


def degrade_image(
    hr_image: np.ndarray,
    scale_factor: int,
    blur_kernel: int = 5,
    blur_sigma_range: tuple[float, float] = (0.2, 2.0),
    noise_std_range: tuple[float, float] = (0.0, 0.05),
    jpeg_quality_range: tuple[int, int] = (60, 100),
    apply_jpeg: bool = True,
) -> np.ndarray:
    """
    Full degradation pipeline: blur -> downsample -> noise -> JPEG compression.

    Args:
        hr_image: HR image as np.ndarray, shape (H, W, 3), dtype uint8, BGR (OpenCV convention).
        scale_factor: integer downscale factor (e.g., 2 or 4).
        blur_kernel: base kernel size for Gaussian blur.
        blur_sigma_range: (min, max) sigma, randomly sampled per call.
        noise_std_range: (min, max) noise std on [0,1] scale, randomly sampled per call.
        jpeg_quality_range: (min, max) JPEG quality, randomly sampled per call.
        apply_jpeg: whether to apply JPEG compression simulation.

    Returns:
        LR image, np.ndarray, shape (H/scale_factor, W/scale_factor, 3), dtype uint8.
    """
    if hr_image is None or hr_image.size == 0:
        raise ValueError("Received an empty or None HR image.")

    sigma = random.uniform(*blur_sigma_range)
    noise_std = random.uniform(*noise_std_range)
    jpeg_quality = random.randint(*jpeg_quality_range)

    img = apply_gaussian_blur(hr_image, blur_kernel, sigma)
    img = downsample(img, scale_factor)
    img = add_gaussian_noise(img, noise_std)

    if apply_jpeg:
        img = apply_jpeg_compression(img, jpeg_quality)

    return img


if __name__ == "__main__":
    # Quick smoke test with a synthetic dummy image (no dataset required yet).
    dummy_hr = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    lr = degrade_image(dummy_hr, scale_factor=4)
    print("HR shape:", dummy_hr.shape)
    print("LR shape:", lr.shape)
    assert lr.shape == (64, 64, 3), "Unexpected LR shape -- check downsample logic."
    print("Smoke test passed.")
