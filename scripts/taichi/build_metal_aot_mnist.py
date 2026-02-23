#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import taichi as ti


NUM_CLASSES = 10
NUM_FEATURES = 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Taichi MNIST (logistic regression) kernels as Metal AOT artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/mnist_aot"),
        help="AOT output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / ".ti-cache"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ti.init(arch=ti.metal, offline_cache_file_path=str(cache_dir))

    x = ti.field(dtype=ti.f32, shape=(NUM_FEATURES,))
    label = ti.field(dtype=ti.i32, shape=())
    w = ti.field(dtype=ti.f32, shape=(NUM_CLASSES, NUM_FEATURES))
    b = ti.field(dtype=ti.f32, shape=(NUM_CLASSES,))
    logits = ti.field(dtype=ti.f32, shape=(NUM_CLASSES,))
    grad_w = ti.field(dtype=ti.f32, shape=(NUM_CLASSES, NUM_FEATURES))
    grad_b = ti.field(dtype=ti.f32, shape=(NUM_CLASSES,))
    loss = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def mnist_init_params():
        for c, j in w:
            v = ti.cast((c * 131 + j * 17) % 97, ti.f32) - 48.0
            w[c, j] = v * 0.0002
        for c in b:
            b[c] = 0.0

    @ti.kernel
    def mnist_forward():
        loss[None] = 0.0
        for c in range(NUM_CLASSES):
            acc = b[c]
            for j in range(NUM_FEATURES):
                acc += w[c, j] * x[j]
            logits[c] = acc
            target = 1.0 if c == label[None] else 0.0
            diff = acc - target
            ti.atomic_add(loss[None], 0.5 * diff * diff)

    @ti.kernel
    def mnist_backward():
        for c in range(NUM_CLASSES):
            target = 1.0 if c == label[None] else 0.0
            g = logits[c] - target
            grad_b[c] = g
            for j in range(NUM_FEATURES):
                grad_w[c, j] = g * x[j]

    @ti.kernel
    def mnist_apply_grad(lr: ti.f32):
        for c, j in w:
            w[c, j] -= lr * grad_w[c, j]
        for c in b:
            b[c] -= lr * grad_b[c]

    module = ti.aot.Module(arch=ti.metal)
    module.add_field("x", x)
    module.add_field("label", label)
    module.add_field("w", w)
    module.add_field("b", b)
    module.add_field("logits", logits)
    module.add_field("grad_w", grad_w)
    module.add_field("grad_b", grad_b)
    module.add_field("loss", loss)
    module.add_kernel(mnist_init_params)
    module.add_kernel(mnist_forward)
    module.add_kernel(mnist_backward)
    module.add_kernel(mnist_apply_grad)
    module.save(str(output_dir))

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    exported = [k["name"] for k in metadata.get("kernels", [])]
    print(f"saved AOT to: {output_dir}")
    print("exported kernels:", ", ".join(exported))


if __name__ == "__main__":
    main()
