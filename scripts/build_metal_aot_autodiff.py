#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import taichi as ti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Taichi autodiff forward/backward kernels as Metal AOT artifacts."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="1D field size used by kernels.",
    )
    parser.add_argument(
        "--base",
        type=float,
        default=0.25,
        help="Base value for x[i] initialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/metal_aot_autodiff"),
        help="AOT output directory.",
    )
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Run runtime gradient correctness check before exporting AOT.",
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
    output_dir.mkdir(parents=True, exist_ok=True)

    ti.init(arch=ti.metal, offline_cache_file_path=str(cache_dir))
    ti.root.lazy_grad()

    x = ti.field(dtype=ti.f32, shape=(args.size,), needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def init_x(base: ti.f32):
        for i in x:
            x[i] = base + 0.01 * i

    @ti.kernel
    def clear_loss():
        loss[None] = 0.0

    @ti.kernel
    def forward():
        # Keep the body loop-only so reverse-mode autodiff can transform it.
        for i in x:
            ti.atomic_add(loss[None], x[i] * x[i])

    if args.runtime_check:
        init_x(args.base)
        clear_loss()
        with ti.ad.Tape(loss=loss):
            forward()
        max_abs_err = 0.0
        for i in range(args.size):
            expected = 2.0 * (args.base + 0.01 * i)
            got = float(x.grad[i])
            max_abs_err = max(max_abs_err, abs(got - expected))
        print(f"runtime grad check: max_abs_err={max_abs_err:.8f}")

    module = ti.aot.Module(arch=ti.metal)
    module.add_field("x", x)
    module.add_field("loss", loss)
    module.add_kernel(init_x)
    module.add_kernel(clear_loss)
    module.add_kernel(forward, name="forward")

    # Taichi's public AOT API exports only primal kernels.
    # We compile and register the reverse-mode kernel via the internal builder.
    forward.grad.ensure_compiled()
    module._aot_builder.add("backward", forward.grad.kernel_cpp)
    module._content.append("kernel:backward")

    module.save(str(output_dir))
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    exported = [k["name"] for k in metadata.get("kernels", [])]

    print(f"saved AOT to: {output_dir}")
    print("exported kernels:", ", ".join(exported))
    print("artifact files:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"- {p.name}")


if __name__ == "__main__":
    main()
