# ldm

This README documents only `/home/qsw26/smartfall/gait_loss/ldm`.

## Structure

- `train.py`: stage-wise training entrypoint.
- `generate.py`: conditional generation entrypoint.
- `diffusion_model/`: dataset, model, graph, diffusion, and checkpoint utilities.
- `checkpoints/`: saved checkpoints.
- `outputs/`: generated outputs (for example GIFs).

## Implemented 3-stage pipeline

Defined in `diffusion_model/model.py`:

1. `Stage1Model`
- Skeleton latent diffusion pretraining.
- Input: skeleton `[B,T,J,3]`.
- Core loss: `loss_diff`.

2. `Stage2Model`
- IMU-to-latent alignment with frozen stage-1 encoder.
- Inputs: `A_hip` and `A_wrist`, each `[B,T,3]`.
- Core loss: `loss_align`.

3. `Stage3Model`
- Conditional latent diffusion + skeleton classifier.
- Core losses: `loss_diff`, `loss_cls`, `loss_total`.
- Formula: `loss_total = loss_diff + lambda_cls * loss_cls`.

Default constants from `diffusion_model/util.py`:

- `timesteps=500`
- `latent_dim=256`
- `window=90`
- `joints=32`
- `num_classes=14`
- `lambda_cls=0.1`

## Data loading modes

Implemented in `diffusion_model/dataset.py`:

1. Torch-file mode (`--dataset_path`)
- Expects tensors from a `.pt`/`.pth` payload.
- Uses keys `skeleton`, `A_hip`/`A`, `A_wrist`/`Omega`, `label`, `fps`, `joint_labels`.

2. CSV-folder mode
- Requires all three flags together:
  - `--skeleton_folder`
  - `--hip_folder`
  - `--wrist_folder`
- Builds sliding windows using `window` and `stride`.

3. Synthetic fallback mode
- Used when neither torch-file nor CSV-folder inputs are provided.

Batch keys used by training/generation:

- `skeleton`
- `A_hip`
- `A_wrist`
- `label`
- `fps`
- `joint_labels`

## Training

Run from `ldm/`:

```bash
cd /home/qsw26/smartfall/gait_loss/ldm
```

Stage 1:

```bash
python train.py --stage 1 --save_dir checkpoints
```

Stage 2:

```bash
python train.py --stage 2 --stage1_ckpt checkpoints/stage1.pt --save_dir checkpoints
```

Stage 3:

```bash
python train.py --stage 3 --stage1_ckpt checkpoints/stage1.pt --stage2_ckpt checkpoints/stage2.pt --save_dir checkpoints
```

Saved files per stage:

- Stage 1: `stage1_best.pt`, `stage1.pt`
- Stage 2: `stage2_best.pt`, `stage2.pt`
- Stage 3: `stage3_best.pt`, `stage3.pt`

Runtime behavior in current `train.py`:

- Rejects multi-process launch when `WORLD_SIZE > 1`.
- `--ddp` is accepted but ignored in single-process mode.
- Uses AMP on CUDA unless `--no_amp`.
- Dataloader uses `drop_last=True`.

## Generation

Run from `ldm/`:

```bash
python generate.py \
  --stage1_ckpt checkpoints/stage1.pt \
  --stage2_ckpt checkpoints/stage2.pt \
  --stage3_ckpt checkpoints/stage3.pt
```

Optional generation flags implemented in `generate.py`:

- `--save_gif`
- `--gif_dir` (default `outputs/gifs`)
- `--gif_prefix` (default `sample`)
- `--gif_fps` (default `12`)
- `--gif_index` (default `0`)
- `--target_class` (formats: `A12`, `12`, or zero-based id)
- `--max_attempts` (default `64`)
- `--h_none`
- `--classify`

## Imported dependencies

From `ldm` source imports:

- `torch`
- `numpy`
- `pandas`
- `Pillow` (`PIL`)
- `tqdm` (optional)
- `torch_geometric` (required by `diffusion_model/graph_modules.py`)
