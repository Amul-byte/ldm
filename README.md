# 3-Stage Joint-Aware Latent Diffusion

Implementation of the strict proposal-aligned pipeline:

1. Stage 1: Skeleton latent diffusion
2. Stage 2: IMU to latent alignment (hip + wrist, accel + gyro)
3. Stage 3: Conditional diffusion and classification

## Structure

```
diffusion_model/
  __init__.py
  dataset.py
  diffusion.py
  graph_modules.py
  model.py
  model_loader.py
  sensor_model.py
  skeleton_model.py
  util.py
README.md
generate.py
train.py
```

## Synthetic Fallback

If `--dataset_path` is omitted, synthetic tensors are generated automatically.

## Train

```bash
python train.py --stage 1
python train.py --stage 2 --stage1_ckpt checkpoints/stage1.pt
python train.py --stage 3 --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt
```

## Generate

```bash
python generate.py --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --classify
```

Use unconditional stage-3 path (`h=None`):

```bash
python train.py --stage 3 --use_h_none
python generate.py --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --h_none
```
