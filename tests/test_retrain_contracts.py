from __future__ import annotations

import pytest
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_model.dataset import TorchFileGaitDataset, _default_gait_cache_dir, _parse_subject_id, _windowed, split_train_val_dataset
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, GAIT_CACHE_VERSION, GAIT_METRIC_NAMES, rotate_and_align_torch
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.sensor_model import IMU_FEATURE_NAMES, build_imu_features


def _dummy_batch(batch_size: int = 2, frames: int = 32, joints: int = 32):
    x = torch.randn(batch_size, frames, joints, 3)
    a_hip = torch.randn(batch_size, frames, 3)
    a_wrist = torch.randn(batch_size, frames, 3)
    gait = torch.randn(batch_size, DEFAULT_GAIT_METRICS_DIM)
    y = torch.randint(0, 14, (batch_size,))
    return x, a_hip, a_wrist, gait, y


def test_gait_metric_contract_is_exact_9_metrics():
    assert DEFAULT_GAIT_METRICS_DIM == 9
    assert len(GAIT_METRIC_NAMES) == 9
    assert "Mean Stride Length" not in GAIT_METRIC_NAMES
    assert GAIT_METRIC_NAMES[-1] == "Mean Base of Support"


def test_gait_cache_dir_is_versioned():
    cache_dir = _default_gait_cache_dir(dataset_path="/tmp/mock.pt", skeleton_folder=None)
    assert GAIT_CACHE_VERSION in cache_dir


def test_windowing_count_for_overlap():
    arr = np.arange(10)
    windows = _windowed(arr, window=4, stride=3)
    assert len(windows) == 3


def test_subject_id_parser_reads_smartfall_style_names():
    assert _parse_subject_id("S29A08T01.csv") == 29


class _SubjectDummyDataset(Dataset):
    def __init__(self):
        self.subject_ids = torch.tensor([28, 28, 29, 30, 30], dtype=torch.long)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        return idx


def test_subject_wise_split_keeps_subjects_disjoint():
    dataset = _SubjectDummyDataset()
    train_subset, val_subset = split_train_val_dataset(dataset, val_split=0.1, seed=42, train_subjects=[28, 30])
    train_subjects = {int(dataset.subject_ids[idx]) for idx in train_subset.indices}
    val_subjects = {int(dataset.subject_ids[idx]) for idx in val_subset.indices}
    assert train_subjects == {28, 30}
    assert val_subjects == {29}
    assert train_subjects.isdisjoint(val_subjects)


def test_torchfile_dataset_raises_when_label_is_missing():
    payload = {
        "skeleton": torch.randn(2, 32, 32, 3),
        "A_hip": torch.randn(2, 32, 3),
        "A_wrist": torch.randn(2, 32, 3),
        "gait_metrics": torch.randn(2, DEFAULT_GAIT_METRICS_DIM),
        "fps": 30,
        "joint_labels": tuple(str(i) for i in range(32)),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(payload, handle.name)
        with pytest.raises(ValueError, match="Missing required 'label'"):
            TorchFileGaitDataset(handle.name, window=32, joints=32)


def test_imu_feature_builder_outputs_expected_channels_and_finite_angles():
    accel = torch.tensor([[[1.0, 2.0, 3.0], [0.5, -0.5, 0.25]]], dtype=torch.float32)
    feats = build_imu_features(accel)
    assert feats.shape == (1, 2, len(IMU_FEATURE_NAMES))
    assert torch.allclose(feats[..., :3], accel)
    assert torch.all(feats[..., 3] >= 0)
    assert torch.isfinite(feats).all()


def test_stage_forward_smoke_losses_are_finite():
    x, a_hip, a_wrist, gait, y = _dummy_batch()

    stage1 = Stage1Model(latent_dim=32, num_joints=32, timesteps=10, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    out1 = stage1(x, gait_metrics=gait)
    assert torch.isfinite(out1["loss_diff"])

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=32, num_joints=32, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    out2 = stage2(x=x, a_hip_stream=a_hip, a_wrist_stream=a_wrist, gait_metrics=gait)
    assert torch.isfinite(out2["loss_align"])

    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_dim=32,
        num_joints=32,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
    )
    out3 = stage3(x=x, y=y, h_tokens=out2["h_tokens"], h_global=out2["h_global"], gait_metrics=gait, fps=30.0)
    for key in ["loss_diff", "loss_cls", "loss_motion", "loss_bone", "loss_skate", "loss_smooth", "loss_instab"]:
        assert torch.isfinite(out3[key]), key
    assert out3["gait_gen"].shape == (x.shape[0], DEFAULT_GAIT_METRICS_DIM)


def test_rotate_and_align_torch_handles_degenerate_sequences():
    pose = torch.zeros(90, 32, 3)
    aligned = rotate_and_align_torch(pose)
    assert aligned.shape == (90, 16, 3)
    assert torch.isfinite(aligned).all()
