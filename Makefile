UV ?= uv
UV_CACHE_DIR ?= .uv-cache
PYTHON ?= .venv/bin/python
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

.PHONY: help setup venv install build build-check metal-src metal-lib ios-shaders
.PHONY: prepare-mnist build-mnist-aot metal-lib-mnist ios-mnist-assets

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

setup: install

venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		UV_CACHE_DIR="$(UV_CACHE_DIR)" "$(UV)" venv --python 3.13 .venv; \
	fi

install: venv pyproject.toml
	UV_CACHE_DIR="$(UV_CACHE_DIR)" "$(UV)" pip install --python "$(PYTHON)" -e .

build: install
	"$(PYTHON)" scripts/build_metal_aot_autodiff.py --size "$(SIZE)" --base "$(BASE)" --output-dir "$(OUTPUT_DIR)"

build-check: install
	"$(PYTHON)" scripts/build_metal_aot_autodiff.py --size "$(SIZE)" --base "$(BASE)" --output-dir "$(OUTPUT_DIR)" --runtime-check

metal-src: build
	"$(PYTHON)" scripts/export_metal_from_spv.py --aot-dir "$(AOT_DIR)" --output-dir "$(METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)"

metal-lib: build
	rm -rf "$(METAL_OUTPUT_DIR)"
	"$(PYTHON)" scripts/export_metal_from_spv.py --aot-dir "$(AOT_DIR)" --output-dir "$(METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)" --compile-metallib

ios-shaders: metal-lib
	mkdir -p "$(IOS_SHADER_DIR)"
	cp "$(METAL_OUTPUT_DIR)/init_x.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/clear_loss.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/forward.metallib" "$(IOS_SHADER_DIR)/"
	cp "$(METAL_OUTPUT_DIR)/backward.metallib" "$(IOS_SHADER_DIR)/"

prepare-mnist: install
	"$(PYTHON)" scripts/prepare_mnist_subset.py --train-count "$(MNIST_TRAIN_COUNT)" --test-count "$(MNIST_TEST_COUNT)" --output "$(IOS_MNIST_DATA_DIR)/mnist_subset.bin"

build-mnist-aot: install
	"$(PYTHON)" scripts/build_metal_aot_mnist.py --output-dir "$(MNIST_AOT_DIR)"

metal-lib-mnist: build-mnist-aot
	rm -rf "$(MNIST_METAL_OUTPUT_DIR)"
	"$(PYTHON)" scripts/export_metal_from_spv.py --aot-dir "$(MNIST_AOT_DIR)" --output-dir "$(MNIST_METAL_OUTPUT_DIR)" --spirv-cross "$(SPIRV_CROSS)" --compile-metallib --preserve-task-names

ios-mnist-assets: prepare-mnist metal-lib-mnist
	mkdir -p "$(IOS_MNIST_SHADER_DIR)"
	mkdir -p "$(IOS_MNIST_DATA_DIR)"
	rm -f "$(IOS_MNIST_SHADER_DIR)"/mnist_*.metallib
	cp "$(MNIST_METAL_OUTPUT_DIR)"/mnist_*.metallib "$(IOS_MNIST_SHADER_DIR)/"
	cp "$(MNIST_AOT_DIR)/metadata.json" "$(IOS_MNIST_DATA_DIR)/mnist_metadata.json"
