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
- Inputs: raw `A_hip` and `A_wrist`, each `[B,T,3]`, expanded internally to `[ax, ay, az, |a|, pitch, roll]`, plus auto-computed `gait_metrics` `[B,G]`.
- Core loss: `loss_align`.

3. `Stage3Model`
- Conditional latent diffusion + skeleton classifier.
- Core losses: `loss_diff`, `loss_cls`, `loss_gait`, `loss_motion`, `loss_total`.
- Formula: `loss_total = loss_diff + lambda_cls * loss_cls + lambda_motion * loss_motion + lambda_gait * loss_gait`.

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
- Optional extra key: `gait_metrics` with shape `[N,G]` or `[G]`; if absent the code computes it automatically from skeletons.

2. CSV-folder mode
- Requires all three flags together:
  - `--skeleton_folder`
  - `--hip_folder`
  - `--wrist_folder`
- Builds sliding windows using `window` and `stride`.
- The loader auto-computes one gait-summary vector per source file and reuses it for all windows from that file.

3. Synthetic fallback mode
- Used when neither torch-file nor CSV-folder inputs are provided.

Batch keys used by training/generation:

- `skeleton`
- `A_hip`
- `A_wrist`
- `gait_metrics`
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
- Gait metrics are auto-computed and cached unless `--disable_gait_cache` is set.

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
- `--gait_cache_dir`
- `--disable_gait_cache`
- `--gait_metrics_dim`
- `--h_none`
- `--classify`

## Auto-computed gait metrics

The code now computes gait metrics internally from the input skeleton using the notebook-style pipeline:

- extract the fixed 16-joint subset from the 32-joint skeleton
- align the skeleton to the ground plane
- detect gait events from smoothed head-height peaks
- compute the gait summary vector
- cache one CSV per source file under `--gait_cache_dir`

The canonical gait vector now uses the exact 9 requested metrics:

- Mean CoM Fore-Aft
- StDev CoM Fore-Aft
- Mean CoM Width
- StDev CoM Width
- Mean CoM Height
- StDev CoM Height
- Mean Walking Speed
- Mean Stride Width
- Mean Base of Support

The same gait-summary vector is used for:

- encoder conditioning
- denoiser conditioning
- Stage-2 IMU alignment
- Stage-3 gait-summary matching loss on generated skeletons

Example:

```bash
python train.py \
  --stage 3 \
  --stage1_ckpt checkpoints/stage1.pt \
  --stage2_ckpt checkpoints/stage2.pt \
  --skeleton_folder /path/to/skeleton \
  --hip_folder /path/to/hip \
  --wrist_folder /path/to/wrist \
  --gait_cache_dir outputs/gait_cache \
  --lambda-gait 0.1 \
  --save_dir checkpoints
```

## Imported dependencies

From `ldm` source imports:

- `torch`
- `numpy`
- `pandas`
- `Pillow` (`PIL`)
- `tqdm` (optional)
- `torch_geometric` (required by `diffusion_model/graph_modules.py`)

## Stage-3 objective

Stage-3 training uses the decoded skeleton `X_gen` to optimize:

- `L_diffusion`: noise-prediction MSE in latent space
- `L_classification`: class CE on decoded skeletons
- `L_gait`: generated-vs-real gait-summary MSE
- `L_motion`: biomechanical regularization (`bone + foot_skating + smoothness + instability`)

Stage-3 optimized total loss is:

`L_total = L_diffusion + lambda_cls * L_classification + lambda_motion * L_motion + lambda_gait * L_gait`

Relevant Stage-3 CLI flags:

- `--lambda_cls` (default `0.1`)
- `--lambda_motion` (default `1.0`)
- `--lambda-gait` (default `1.0`)
- `--fps` (default `30`)
- `--sample_steps` (default `50`)

Example Stage-3 training command:

```bash
python train.py \
  --stage 3 \
  --stage1_ckpt checkpoints/stage1.pt \
  --stage2_ckpt checkpoints/stage2.pt \
  --save_dir checkpoints \
  --lambda_cls 0.1 \
  --lambda_motion 1.0 \
  --lambda-gait 1.0 \
  --fps 30
```
