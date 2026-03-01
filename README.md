# 3-Stage Joint-Aware Latent Diffusion

Implementation aligned to the proposal architecture:

1. Stage 1: Skeleton graph encoder latent diffusion pre-training.
2. Stage 2: Two-branch IMU Temporal-GNN (T-GNN) alignment to pooled skeleton latent target.
3. Stage 3: Conditional latent diffusion with graph denoiser + cross-attention from temporal sensor tokens.

Notes:
- Stage 1 optimization uses diffusion noise loss (`L_diff`) as proposal-exact objective.
- GAT layers are mandatory (requires `torch_geometric`).

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

Quick synthetic training run:

```bash
python train.py --stage 1 --epochs 1 --batch_size 2 --synthetic_length 8 --timesteps 50 --save_dir checkpoints
python train.py --stage 2 --epochs 1 --batch_size 2 --synthetic_length 8 --timesteps 50 --stage1_ckpt checkpoints/stage1.pt --save_dir checkpoints
python train.py --stage 3 --epochs 1 --batch_size 2 --synthetic_length 8 --timesteps 50 --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --save_dir checkpoints
```

## Generate

```bash
python generate.py \
  --stage1_ckpt checkpoints/stage1.pt \
  --stage2_ckpt checkpoints/stage2.pt \
  --stage3_ckpt checkpoints/stage3.pt \
  --classify
```

Generate GIF outputs:

```bash
python generate.py \
  --stage1_ckpt checkpoints/stage1.pt \
  --stage2_ckpt checkpoints/stage2.pt \
  --stage3_ckpt checkpoints/stage3.pt \
  --timesteps 50 \
  --batch_size 2 \
  --save_gif \
  --gif_dir outputs/gifs \
  --gif_prefix gait \
  --classify
```

Use unconditional stage-3 path (`h_tokens=None`, `h_global=None`):

```bash
python train.py --stage 3 --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --use_h_none
python generate.py --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --stage3_ckpt checkpoints/stage3.pt --h_none
```
