# Taichi to Metal AOT Playground (with iOS MNIST demo)

This repository is a practical testbed for:

- exporting Taichi kernels (including autodiff forward/backward) as AOT artifacts
- converting generated SPIR-V shaders to Metal (`.metal` / `.metallib`)
- running those kernels in a minimal iOS app
- training and inferring a simple MNIST model on-device with the same pipeline

## Repository purpose

The main goal is to validate an end-to-end workflow:

1. Author compute kernels in Taichi.
2. Export AOT package (`metadata.json`, task-level `.spv`).
3. Convert `.spv` to Metal via [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross).
4. Bundle `.metallib` files into an iOS app.
5. Execute forward/backward and inspect results in app UI.

## Project structure

- `scripts/build_metal_aot_autodiff.py`: exports a small autodiff example (`forward` / `backward`).
- `scripts/build_metal_aot_mnist.py`: exports MNIST training/inference kernels.
- `scripts/export_metal_from_spv.py`: converts Taichi-generated `.spv` into Metal source and libraries.
- `scripts/prepare_mnist_subset.py`: prepares an app-friendly MNIST subset binary.
- `TaichiJitExampleApp/`: SwiftUI iOS app that runs kernels with Metal.
- `Makefile`: single entrypoint for setup and build flows.

## Requirements

- macOS with Xcode and Metal toolchain
- `uv` for Python environment setup
- `spirv-cross` (install with `brew install spirv-cross`)

## Quick start

```bash
make setup
```

## Autodiff pipeline (Taichi -> Metal)

```bash
# build Taichi AOT (includes forward/backward setup)
make build

# optional runtime gradient sanity check
make build-check

# convert .spv -> .metal
make metal-src

# convert .spv -> .metal -> .metallib
make metal-lib

# copy stable demo shaders into iOS app resources
make ios-shaders
```

## MNIST iOS demo pipeline

```bash
# prepare dataset + build MNIST AOT + compile/copy MNIST metallibs
make ios-mnist-assets

# optional simulator build check
xcodebuild -project TaichiJitExampleApp/TaichiJitExampleApp.xcodeproj \
  -scheme TaichiJitExampleApp \
  -sdk iphonesimulator \
  -configuration Debug \
  -derivedDataPath .derived \
  build
```

In the app:

- tap `Train MNIST` to run on-device training
- tap `Infer Random` to run inference and show the input image + prediction
- adjust epochs / LR / loss-threshold stop in the UI settings

## Config knobs

- autodiff demo:
  - `SIZE`, `BASE`, `OUTPUT_DIR`
- MNIST subset:
  - `MNIST_TRAIN_COUNT`, `MNIST_TEST_COUNT`

Example:

```bash
make ios-mnist-assets MNIST_TRAIN_COUNT=8000 MNIST_TEST_COUNT=1000
```

## Notes

- Taichi Metal AOT emits task-level SPIR-V (`.spv`) first. Metal is generated in a separate conversion step.
- The autodiff backward kernel is registered via Taichi internal builder API (`module._aot_builder`) in this repo.
- If `xcrun metal` reports missing components, install/update Metal toolchain components in Xcode.
