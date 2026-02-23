UV ?= uv
UV_CACHE_DIR ?= .uv-cache
PYTHON ?= .venv/bin/python
TAICHI_SCRIPT_DIR ?= scripts/taichi
SLANG_SCRIPT_DIR ?= scripts/slang
SIZE ?= 16
BASE ?= 0.25
OUTPUT_DIR ?= build/metal_aot_autodiff
AOT_DIR ?= build/metal_aot_autodiff
METAL_OUTPUT_DIR ?= build/metal_shaders
SPIRV_CROSS ?= spirv-cross
IOS_SHADER_DIR ?= TaichiJitExampleApp/TaichiJitExampleApp/Shaders
MNIST_AOT_DIR ?= build/mnist_aot
MNIST_METAL_OUTPUT_DIR ?= build/mnist_shaders
MNIST_TRAIN_COUNT ?= 2000
MNIST_TEST_COUNT ?= 400
IOS_MNIST_SHADER_DIR ?= TaichiJitExampleApp/TaichiJitExampleApp/Shaders/MNIST
IOS_MNIST_DATA_DIR ?= TaichiJitExampleApp/TaichiJitExampleApp/MNIST
SLANGC ?= slangc
SLANG_PROFILE ?= metal_2_4
SLANG_SOURCE ?= slang/probes/autodiff_probe.slang
SLANG_VERIFY_DIR ?= build/slang_verify
SLANG_COMPOSE_FILE ?= slang/compose.yml
SLANG_TO_MLX_METAL ?= build/slang_verify/run_backward_custom.metal
SLANG_TO_MLX_ENTRY ?= run_backward_custom
SLANG_TO_MLX_SWIFT ?= build/slang_verify/run_backward_custom_mlx.swift
SLANG_TO_MLX_JSON ?= build/slang_verify/run_backward_custom_mlx.json
SLANG_TO_MLX_BUNDLE_JSON ?= TaichiJitExampleApp/TaichiJitExampleApp/Slang/run_backward_custom_mlx.json

.PHONY: help setup venv install build build-check metal-src metal-lib ios-shaders
.PHONY: prepare-mnist build-mnist-aot metal-lib-mnist ios-mnist-assets slang-check slang-check-docker slang-to-mlx ios-slang-assets

help:
	@echo "Targets:"
	@echo "  make setup        # create .venv and install dependencies with uv"
	@echo "  make build        # export Metal AOT (forward/backward)"
	@echo "  make build-check  # export Metal AOT + runtime gradient check"
	@echo "  make metal-src    # convert AOT .spv to .metal via SPIRV-Cross"
	@echo "  make metal-lib    # convert AOT .spv to .metal and compile .metallib"
	@echo "  make ios-shaders  # copy stable .metallib files into iOS app bundle resources"
	@echo "  make prepare-mnist    # download MNIST subset for the iOS app"
	@echo "  make build-mnist-aot  # build MNIST kernels as Taichi AOT"
	@echo "  make metal-lib-mnist  # convert MNIST AOT kernels to .metallib"
	@echo "  make ios-mnist-assets # prepare MNIST data + shaders for iOS app"
	@echo "  make slang-check      # compile Slang autodiff sample to Metal/.metallib"
	@echo "  make slang-check-docker # run Slang probe in Docker and emit .metal files"
	@echo "  make slang-to-mlx     # auto-convert Slang .metal into MLX metalKernel Swift snippet"
	@echo "  make ios-slang-assets # generate bundle JSON for MNIST MLX Slang feature kernel"

setup: install

venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		UV_CACHE_DIR="$(UV_CACHE_DIR)" "$(UV)" venv --python 3.13 .venv; \
	fi

install: venv pyproject.toml
	UV_CACHE_DIR="$(UV_CACHE_DIR)" "$(UV)" pip install --python "$(PYTHON)" -e .

build: install
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/build_metal_aot_autodiff.py" --size "$(SIZE)" --base "$(BASE)" --output-dir "$(OUTPUT_DIR)"

build-check: install
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/build_metal_aot_autodiff.py" --size "$(SIZE)" --base "$(BASE)" --output-dir "$(OUTPUT_DIR)" --runtime-check

metal-src: build
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/export_metal_from_spv.py" --aot-dir "$(AOT_DIR)" --output-dir "$(METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)"

metal-lib: build
	rm -rf "$(METAL_OUTPUT_DIR)"
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/export_metal_from_spv.py" --aot-dir "$(AOT_DIR)" --output-dir "$(METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)" --compile-metallib

ios-shaders: metal-lib
	mkdir -p "$(IOS_SHADER_DIR)"
	cp "$(METAL_OUTPUT_DIR)/init_x.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/clear_loss.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/forward.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/backward.metallib" "$(IOS_SHADER_DIR)/"

prepare-mnist: install
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/prepare_mnist_subset.py" --train-count "$(MNIST_TRAIN_COUNT)" --test-count "$(MNIST_TEST_COUNT)" --output "$(IOS_MNIST_DATA_DIR)/mnist_subset.bin"

build-mnist-aot: install
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/build_metal_aot_mnist.py" --output-dir "$(MNIST_AOT_DIR)"

metal-lib-mnist: build-mnist-aot
	rm -rf "$(MNIST_METAL_OUTPUT_DIR)"
	"$(PYTHON)" "$(TAICHI_SCRIPT_DIR)/export_metal_from_spv.py" --aot-dir "$(MNIST_AOT_DIR)" --output-dir "$(MNIST_METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)" --compile-metallib --preserve-task-names

ios-mnist-assets: prepare-mnist metal-lib-mnist
	mkdir -p "$(IOS_MNIST_SHADER_DIR)"
	mkdir -p "$(IOS_MNIST_DATA_DIR)"
	rm -f "$(IOS_MNIST_SHADER_DIR)"/mnist_*.metallib
	cp "$(MNIST_METAL_OUTPUT_DIR)"/mnist_*.metallib "$(IOS_MNIST_SHADER_DIR)/"
	cp "$(MNIST_AOT_DIR)/metadata.json" "$(IOS_MNIST_DATA_DIR)/mnist_metadata.json"

slang-check:
	SLANGC="$(SLANGC)" \
	SLANG_PROFILE="$(SLANG_PROFILE)" \
	SLANG_SOURCE="$(SLANG_SOURCE)" \
	SLANG_OUT_DIR="$(SLANG_VERIFY_DIR)" \
	bash "$(SLANG_SCRIPT_DIR)/verify_slang_autodiff.sh"

slang-check-docker:
	docker compose -f "$(SLANG_COMPOSE_FILE)" build slang
	docker compose -f "$(SLANG_COMPOSE_FILE)" run --rm slang

slang-to-mlx:
	"$(PYTHON)" "$(SLANG_SCRIPT_DIR)/convert_slang_metal_to_mlx.py" \
		--metal "$(SLANG_TO_MLX_METAL)" \
		--entry "$(SLANG_TO_MLX_ENTRY)" \
		--swift-out "$(SLANG_TO_MLX_SWIFT)" \
		--json-out "$(SLANG_TO_MLX_JSON)"

ios-slang-assets:
	mkdir -p "$$(dirname "$(SLANG_TO_MLX_BUNDLE_JSON)")"
	"$(PYTHON)" "$(SLANG_SCRIPT_DIR)/convert_slang_metal_to_mlx.py" \
		--metal "$(SLANG_TO_MLX_METAL)" \
		--entry "$(SLANG_TO_MLX_ENTRY)" \
		--kernel-name "mnist_slang_feature_transform" \
		--swift-out "$(SLANG_TO_MLX_SWIFT)" \
		--json-out "$(SLANG_TO_MLX_BUNDLE_JSON)"
