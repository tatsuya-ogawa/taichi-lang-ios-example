# Pipelines

This project keeps workflow scripts split by domain:

- `scripts/taichi`: Taichi AOT build/export flow
- `scripts/slang`: Slang compile/convert flow

## Taichi pipeline

Use this when building AOT kernels from Taichi and converting to Metal:

```bash
make build
make metal-lib
make ios-shaders
```

For MNIST assets:

```bash
make ios-mnist-assets
```

## Slang pipeline

Use this when validating Slang autodiff output and generating MLX kernel specs:

```bash
make slang-check
make slang-to-mlx
make ios-slang-assets
```

Docker-based Slang validation:

```bash
make slang-check-docker
```

Related files:

- `slang/probes/autodiff_probe.slang`
- `slang/compose.yml`
- `slang/Dockerfile`
