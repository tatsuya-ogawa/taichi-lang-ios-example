#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def check_spirv_cross(path: str) -> str:
    resolved = shutil.which(path) if "/" not in path else path
    if not resolved or not Path(resolved).exists():
        raise SystemExit(
            "spirv-cross not found. Install it first, e.g. `brew install spirv-cross`."
        )
    return resolved


def metal_toolchain_ready() -> tuple[bool, str]:
    if not shutil.which("xcrun"):
        return False, "xcrun not found."
    probe = subprocess.run(
        ["xcrun", "metal", "-v"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return True, ""
    msg = (probe.stdout or "") + (probe.stderr or "")
    if "missing Metal Toolchain" in msg:
        return (
            False,
            "Metal Toolchain missing. Run `xcodebuild -downloadComponent MetalToolchain`.",
        )
    return False, msg.strip() or "xcrun metal is not available."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Taichi AOT .spv kernels to Metal source (.metal) and optional .metallib."
    )
    parser.add_argument(
        "--aot-dir",
        type=Path,
        default=Path("build/metal_aot_autodiff"),
        help="Input Taichi AOT directory containing .spv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/metal_shaders"),
        help="Output directory for .metal/.air/.metallib.",
    )
    parser.add_argument(
        "--spirv-cross",
        default="spirv-cross",
        help="Path or command name for SPIRV-Cross CLI.",
    )
    parser.add_argument(
        "--compile-metallib",
        action="store_true",
        help="Also compile .metal into .air and .metallib with xcrun.",
    )
    parser.add_argument(
        "--preserve-task-names",
        action="store_true",
        help="Keep output file names as task names (*.spv stem) instead of merging by kernel name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    aot_dir = args.aot_dir
    if not aot_dir.is_absolute():
        aot_dir = project_root / aot_dir

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aot_dir.exists():
        raise SystemExit(f"AOT directory not found: {aot_dir}")

    spirv_cross = check_spirv_cross(args.spirv_cross)

    spv_files = sorted(aot_dir.glob("*.spv"))
    if not spv_files:
        raise SystemExit(f"No .spv files found in: {aot_dir}")

    task_to_kernel: dict[str, str] = {}
    metadata_path = aot_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            for kernel in metadata.get("kernels", []):
                kernel_name = kernel.get("name")
                tasks = kernel.get("tasks_attribs", [])
                if not kernel_name or not tasks:
                    continue
                task_name = tasks[0].get("name")
                if task_name:
                    task_to_kernel[task_name] = kernel_name
        except Exception:
            task_to_kernel = {}

    can_compile = False
    metal_toolchain_reason = ""
    if args.compile_metallib:
        can_compile, metal_toolchain_reason = metal_toolchain_ready()
        if not can_compile:
            raise SystemExit(f"Cannot compile .metallib: {metal_toolchain_reason}")

    manifest: list[dict[str, str]] = []

    for spv in spv_files:
        kernel_name = task_to_kernel.get(spv.stem, spv.stem)
        logical_name = spv.stem if args.preserve_task_names else kernel_name
        metal = output_dir / f"{logical_name}.metal"
        run([spirv_cross, str(spv), "--msl", "--stage", "comp", "--output", str(metal)])

        row = {
            "kernel": kernel_name,
            "task": spv.stem,
            "spv": str(spv),
            "metal": str(metal),
        }

        if can_compile:
            air = output_dir / f"{logical_name}.air"
            metallib = output_dir / f"{logical_name}.metallib"
            run(["xcrun", "metal", "-c", str(metal), "-o", str(air)])
            run(["xcrun", "metallib", str(air), "-o", str(metallib)])
            row["air"] = str(air)
            row["metallib"] = str(metallib)

        manifest.append(row)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"converted {len(spv_files)} shaders")
    print(f"input: {aot_dir}")
    print(f"output: {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
