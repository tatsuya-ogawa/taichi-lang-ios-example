#!/usr/bin/env python3

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MNIST and export an app-friendly binary subset."
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=2000,
        help="Number of train samples to export.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=400,
        help="Number of test samples to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset sampling.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(".cache/mnist"),
        help="Directory for downloaded MNIST files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TaichiJitExampleApp/TaichiJitExampleApp/MNIST/mnist_subset.bin"),
        help="Output binary path.",
    )
    return parser.parse_args()


def ensure_file(path: Path, file_name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    out = path / file_name
    if not out.exists():
        url = f"{MNIST_BASE_URL}/{file_name}"
        print(f"downloading: {url}")
        urllib.request.urlretrieve(url, out)
    return out


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"unexpected image magic number: {magic}")
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(num, rows * cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"unexpected label magic number: {magic}")
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.shape[0] != num:
        raise RuntimeError("label length mismatch")
    return arr


def choose_subset(
    images: np.ndarray,
    labels: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count > images.shape[0]:
        raise RuntimeError(f"requested count {count} exceeds available {images.shape[0]}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(images.shape[0], size=count, replace=False)
    return images[idx], labels[idx]


def write_binary(
    path: Path,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_size = train_images.shape[1]

    train_images_f = (train_images.astype(np.float32) / 255.0).astype("<f4")
    test_images_f = (test_images.astype(np.float32) / 255.0).astype("<f4")
    train_labels_u8 = train_labels.astype(np.uint8)
    test_labels_u8 = test_labels.astype(np.uint8)

    with path.open("wb") as f:
        f.write(struct.pack("<4sIIII", b"MNST", 1, image_size, train_images_f.shape[0], test_images_f.shape[0]))
        f.write(train_images_f.tobytes(order="C"))
        f.write(train_labels_u8.tobytes(order="C"))
        f.write(test_images_f.tobytes(order="C"))
        f.write(test_labels_u8.tobytes(order="C"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    download_dir = args.download_dir
    output = args.output
    if not download_dir.is_absolute():
        download_dir = root / download_dir
    if not output.is_absolute():
        output = root / output

    train_images_path = ensure_file(download_dir, MNIST_FILES["train_images"])
    train_labels_path = ensure_file(download_dir, MNIST_FILES["train_labels"])
    test_images_path = ensure_file(download_dir, MNIST_FILES["test_images"])
    test_labels_path = ensure_file(download_dir, MNIST_FILES["test_labels"])

    train_images_all = read_idx_images(train_images_path)
    train_labels_all = read_idx_labels(train_labels_path)
    test_images_all = read_idx_images(test_images_path)
    test_labels_all = read_idx_labels(test_labels_path)

    train_images, train_labels = choose_subset(
        train_images_all, train_labels_all, args.train_count, args.seed
    )
    test_images, test_labels = choose_subset(
        test_images_all, test_labels_all, args.test_count, args.seed + 1
    )

    write_binary(output, train_images, train_labels, test_images, test_labels)
    print(f"wrote subset binary: {output}")
    print(
        f"train={train_images.shape[0]} test={test_images.shape[0]} image_size={train_images.shape[1]}"
    )


if __name__ == "__main__":
    main()
