from __future__ import annotations

import logging
import pytest
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_model.dataset import (
    TorchFileGaitDataset,
    _default_gait_cache_dir,
    _parse_subject_id,
    _windowed,
    split_train_val_dataset,
)
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, GAIT_CACHE_VERSION, GAIT_METRIC_NAMES, rotate_and_align_torch
from diffusion_model.model import Stage1Model, Stage2Model, Stage3Model
from diffusion_model.model_loader import (
    infer_graph_ops_from_checkpoint,
    infer_temporal_block_type_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from diffusion_model.shared_features import build_shared_motion_features, compute_skeleton_acceleration
from diffusion_model.sensor_model import (
    IMU_FEATURE_NAMES,
    IMUGraphEncoder,
    IMULatentAligner,
    SensorGCNEncoder,
    build_imu_features,
    build_imu_graph_adjacency,
)
from diffusion_model.skeleton_model import (
    GraphDecoder,
    GraphDecoderGCN,
    GraphDenoiserMasked,
    GraphDenoiserMaskedGCN,
    GraphEncoder,
    GraphEncoderGCN,
)
from diffusion_model.training_eval import (
    evaluate_stage1,
    evaluate_stage2_reports,
    evaluate_stage3,
    plot_per_class_accuracy,
    sample_stage3_latents,
    write_classification_artifacts,
    write_history,
)
from diffusion_model.util import DEFAULT_JOINTS, SOURCE_JOINTS_32, get_joint_labels, get_source_joint_labels_32


def _dummy_batch(batch_size: int = 2, frames: int = 32, joints: int = DEFAULT_JOINTS):
    x = torch.randn(batch_size, frames, joints, 3)
    a_hip = torch.randn(batch_size, frames, 3)
    a_wrist = torch.randn(batch_size, frames, 3)
    gait = torch.randn(batch_size, DEFAULT_GAIT_METRICS_DIM)
    y = torch.randint(0, 14, (batch_size,))
    return x, a_hip, a_wrist, gait, y


class _EvalDummyDataset(Dataset):
    def __init__(self, length: int = 4, frames: int = 8, joints: int = DEFAULT_JOINTS):
        self.length = length
        self.frames = frames
        self.joints = joints

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = idx % 14
        return {
            "skeleton": torch.randn(self.frames, self.joints, 3),
            "A_hip": torch.randn(self.frames, 3),
            "A_wrist": torch.randn(self.frames, 3),
            "gait_metrics": torch.randn(DEFAULT_GAIT_METRICS_DIM),
            "label": torch.tensor(label, dtype=torch.long),
            "fps": torch.tensor(30.0),
            "joint_labels": get_joint_labels(),
        }


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


def test_csv_pairing_contract_uses_shortest_shared_length():
    skeleton = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    hip = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    wrist = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)

    t = min(skeleton.shape[0], hip.shape[0], wrist.shape[0])
    aligned_skeleton = skeleton[:t]
    aligned_hip = hip[:t]
    aligned_wrist = wrist[:t]

    assert t == 4
    assert aligned_skeleton.shape == (4, 2, 3)
    assert aligned_hip.shape == (4, 3)
    assert aligned_wrist.shape == (4, 3)
    assert np.array_equal(aligned_hip, hip[:4])
    assert np.array_equal(aligned_wrist, wrist[:4])


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
        "skeleton": torch.randn(2, 32, SOURCE_JOINTS_32, 3),
        "A_hip": torch.randn(2, 32, 3),
        "A_wrist": torch.randn(2, 32, 3),
        "gait_metrics": torch.randn(2, DEFAULT_GAIT_METRICS_DIM),
        "fps": 30,
        "joint_labels": get_source_joint_labels_32(),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(payload, handle.name)
        with pytest.raises(ValueError, match="Missing required 'label'"):
            TorchFileGaitDataset(handle.name, window=32, joints=DEFAULT_JOINTS)


def test_torchfile_dataset_projects_legacy_32_joint_payloads_to_canonical_16():
    payload = {
        "skeleton": torch.randn(2, 32, SOURCE_JOINTS_32, 3),
        "A_hip": torch.randn(2, 32, 3),
        "A_wrist": torch.randn(2, 32, 3),
        "gait_metrics": torch.randn(2, DEFAULT_GAIT_METRICS_DIM),
        "label": torch.tensor([1, 2], dtype=torch.long),
        "fps": 30,
        "joint_labels": get_source_joint_labels_32(),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(payload, handle.name)
        dataset = TorchFileGaitDataset(handle.name, window=32, joints=DEFAULT_JOINTS)
    sample = dataset[0]
    assert sample["skeleton"].shape == (32, DEFAULT_JOINTS, 3)
    assert tuple(sample["joint_labels"]) == get_joint_labels()


def test_torchfile_dataset_rejects_noncanonical_requested_joint_count():
    with pytest.raises(ValueError, match="canonical 16-joint layout"):
        TorchFileGaitDataset("/tmp/unused.pt", window=32, joints=SOURCE_JOINTS_32)


def test_imu_feature_builder_outputs_expected_channels_and_finite_angles():
    accel = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.5, -0.5, 0.25], [-1.0, 0.25, 2.0], [0.0, 1.5, -0.75]]],
        dtype=torch.float32,
    )
    feats = build_imu_features(accel)
    assert feats.shape == (1, accel.shape[1], len(IMU_FEATURE_NAMES))
    assert torch.isfinite(feats).all()
    assert torch.allclose(feats[..., :3], accel)
    assert torch.all(feats[..., 3] > 0.0)


def test_build_shared_motion_features_shapes_and_expected_channels():
    accel = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.5, -1.0, 0.5], [0.5, 0.25, -1.5], [-0.5, 1.0, 2.0], [1.5, 3.0, -0.75]]],
        dtype=torch.float32,
    )
    feats_imu = build_shared_motion_features(accel)
    feats_skel = build_shared_motion_features(accel.unsqueeze(2).repeat(1, 1, 2, 1))

    assert feats_imu.shape == (1, accel.shape[1], len(IMU_FEATURE_NAMES))
    assert feats_skel.shape == (1, accel.shape[1], 2, len(IMU_FEATURE_NAMES))
    assert torch.isfinite(feats_imu).all()
    assert torch.isfinite(feats_skel).all()
    assert torch.allclose(feats_imu[..., :3], accel)
    assert torch.allclose(feats_skel[:, :, 0, :], feats_skel[:, :, 1, :])
    expected_mag = torch.linalg.norm(accel, dim=-1)
    assert torch.allclose(feats_imu[..., 3], expected_mag, atol=1e-5)


def test_compute_skeleton_acceleration_exact_shape_and_values():
    positions = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[6.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    accel = compute_skeleton_acceleration(positions)
    expected = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    assert accel.shape == positions.shape
    assert torch.allclose(accel, expected)


def test_skeleton_graph_variants_have_shape_parity():
    x, _, _, _, _ = _dummy_batch(batch_size=2, frames=8, joints=DEFAULT_JOINTS)
    latent_dim = 32
    z = torch.randn(x.shape[0], x.shape[1], x.shape[2], latent_dim)
    t = torch.randint(0, 10, (x.shape[0],), dtype=torch.long)
    h_tokens = torch.randn(x.shape[0], x.shape[1], latent_dim)
    h_global = torch.randn(x.shape[0], latent_dim)

    enc_gat = GraphEncoder(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    enc_gcn = GraphEncoderGCN(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    z_gat = enc_gat(x)
    z_gcn = enc_gcn(x)
    assert z_gat.shape == z_gcn.shape == (x.shape[0], x.shape[1], DEFAULT_JOINTS, latent_dim)

    dec_gat = GraphDecoder(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    dec_gcn = GraphDecoderGCN(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    x_hat_gat = dec_gat(z)
    x_hat_gcn = dec_gcn(z)
    assert x_hat_gat.shape == x_hat_gcn.shape == x.shape

    den_gat = GraphDenoiserMasked(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    den_gcn = GraphDenoiserMaskedGCN(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2)
    eps_gat = den_gat(z, t, h_tokens=h_tokens, h_global=h_global)
    eps_gcn = den_gcn(z, t, h_tokens=h_tokens, h_global=h_global)
    assert eps_gat.shape == eps_gcn.shape == z.shape
    assert torch.isfinite(z_gat).all()
    assert torch.isfinite(z_gcn).all()
    assert torch.isfinite(x_hat_gat).all()
    assert torch.isfinite(x_hat_gcn).all()
    assert torch.isfinite(eps_gat).all()
    assert torch.isfinite(eps_gcn).all()


def test_denoiser_temporal_attention_smoke():
    """Verify temporal attention denoiser produces correct shapes and finite values."""
    latent_dim = 32
    z = torch.randn(2, 8, DEFAULT_JOINTS, latent_dim)
    t = torch.randint(0, 10, (2,), dtype=torch.long)
    h_tokens = torch.randn(2, 8, latent_dim)
    h_global = torch.randn(2, latent_dim)

    den = GraphDenoiserMasked(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2, temporal_block_type="attention")
    eps = den(z, t, h_tokens=h_tokens, h_global=h_global)
    assert eps.shape == z.shape
    assert torch.isfinite(eps).all()

    den_gcn = GraphDenoiserMaskedGCN(latent_dim=latent_dim, num_joints=DEFAULT_JOINTS, depth=2, temporal_block_type="attention")
    eps_gcn = den_gcn(z, t, h_tokens=h_tokens, h_global=h_global)
    assert eps_gcn.shape == z.shape
    assert torch.isfinite(eps_gcn).all()


def test_stage1_temporal_attention_losses_finite():
    """Verify Stage1Model with temporal attention produces finite losses."""
    x, _, _, gait, _ = _dummy_batch()
    stage1 = Stage1Model(
        latent_dim=32, num_joints=DEFAULT_JOINTS, timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        temporal_block_type="attention",
    )
    out = stage1(x, gait_metrics=gait)
    for key in ["loss_diff", "loss_temporal", "loss_joint_corr", "loss_cls", "loss_var"]:
        assert torch.isfinite(out[key]), f"{key} is not finite"


def test_stage_forward_smoke_losses_are_finite():
    x, a_hip, a_wrist, gait, y = _dummy_batch()

    stage1 = Stage1Model(
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    out1 = stage1(x, gait_metrics=gait)
    assert torch.isfinite(out1["loss_diff"])
    assert torch.isfinite(out1["loss_temporal"])
    assert torch.isfinite(out1["loss_joint_corr"])

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=32, num_joints=DEFAULT_JOINTS, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    out2 = stage2(x=x, a_hip_stream=a_hip, a_wrist_stream=a_wrist, gait_metrics=gait)
    assert torch.isfinite(out2["loss_align"])
    assert torch.isfinite(out2["loss_feature"])

    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    out3 = stage3(
        x=x,
        h_tokens=out2["h_tokens"],
        h_global=out2["h_global"],
        gait_target=gait,
        fps=30.0,
        a_hip_stream=a_hip,
        a_wrist_stream=a_wrist,
    )
    for key in [
        "loss_diff",
        "loss_pose",
        "loss_latent",
        "loss_vel",
        "loss_angle",
        "loss_angvel",
        "loss_angle_limit",
        "loss_gait",
        "loss_motion",
        "loss_bone",
        "loss_skate",
        "loss_smooth",
        "loss_instab",
    ]:
        assert torch.isfinite(out3[key]), key
    assert out3["gait_gen"].shape == (x.shape[0], DEFAULT_GAIT_METRICS_DIM)
    assert out3["z0_target"].shape == out3["z0_gen"].shape


def test_stage_models_reject_noncanonical_joint_count():
    with pytest.raises(ValueError, match="canonical 16-joint layout"):
        Stage1Model(latent_dim=16, num_joints=SOURCE_JOINTS_32, timesteps=10, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)


def test_stage_forward_smoke_losses_are_finite_full_gcn():
    x, a_hip, a_wrist, gait, _ = _dummy_batch()

    stage1 = Stage1Model(
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        skeleton_graph_op="gcn",
    )
    assert isinstance(stage1.encoder, GraphEncoderGCN)
    assert isinstance(stage1.decoder, GraphDecoderGCN)
    assert isinstance(stage1.denoiser, GraphDenoiserMaskedGCN)
    out1 = stage1(x, gait_metrics=gait)
    assert torch.isfinite(out1["loss_diff"])
    assert torch.isfinite(out1["loss_temporal"])
    assert torch.isfinite(out1["loss_joint_corr"])

    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=32, num_joints=DEFAULT_JOINTS, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    out2 = stage2(x=x, a_hip_stream=a_hip, a_wrist_stream=a_wrist, gait_metrics=gait)

    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    out3 = stage3(
        x=x,
        h_tokens=out2["h_tokens"],
        h_global=out2["h_global"],
        gait_target=gait,
        fps=30.0,
        a_hip_stream=a_hip,
        a_wrist_stream=a_wrist,
    )
    assert torch.isfinite(out3["loss_diff"])
    assert torch.isfinite(out3["loss_pose"])
    assert torch.isfinite(out3["loss_angle"])
    assert torch.isfinite(out3["loss_angvel"])


def test_stage1_supports_partial_gcn_encoder_override():
    stage1 = Stage1Model(latent_dim=32, skeleton_graph_op="gat", encoder_type="gcn")
    assert isinstance(stage1.encoder, GraphEncoderGCN)
    assert isinstance(stage1.decoder, GraphDecoder)
    assert isinstance(stage1.denoiser, GraphDenoiserMasked)


def test_stage3_sampling_is_deterministic_for_fixed_seed():
    x, a_hip, a_wrist, gait, _ = _dummy_batch(batch_size=1)
    stage1 = Stage1Model(
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=32, num_joints=DEFAULT_JOINTS, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    h_tokens, h_global = stage2.aligner(a_hip, a_wrist)
    shape = torch.Size((1, x.shape[1], x.shape[2], 32))
    # Pass real gait metrics (instead of None) to match the train/eval conditioning contract.
    z_a = sample_stage3_latents(stage3, shape, x.device, h_tokens, h_global, gait_metrics=gait, sample_steps=5, sampler="ddim", sample_seed=7, a_hip_stream=a_hip, a_wrist_stream=a_wrist)
    z_b = sample_stage3_latents(stage3, shape, x.device, h_tokens, h_global, gait_metrics=gait, sample_steps=5, sampler="ddim", sample_seed=7, a_hip_stream=a_hip, a_wrist_stream=a_wrist)
    z_c = sample_stage3_latents(stage3, shape, x.device, h_tokens, h_global, gait_metrics=gait, sample_steps=5, sampler="ddim", sample_seed=8, a_hip_stream=a_hip, a_wrist_stream=a_wrist)
    assert torch.allclose(z_a, z_b)
    assert not torch.allclose(z_a, z_c)


def test_checkpoint_graph_metadata_and_inference():
    stage1 = Stage1Model(latent_dim=32, skeleton_graph_op="gcn")
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        save_checkpoint(
            handle.name,
            stage1,
            extra={
                "encoder_graph_op": stage1.encoder_graph_op,
                "skeleton_graph_op": stage1.skeleton_graph_op,
            },
        )
        encoder_graph_op, skeleton_graph_op = infer_graph_ops_from_checkpoint(handle.name)
    assert encoder_graph_op == "gcn"
    assert skeleton_graph_op == "gcn"


def test_checkpoint_temporal_block_type_inference_from_state_dict():
    stage1 = Stage1Model(latent_dim=32, skeleton_graph_op="gcn", temporal_block_type="attention")
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        save_checkpoint(
            handle.name,
            stage1,
            extra={
                "encoder_graph_op": stage1.encoder_graph_op,
                "skeleton_graph_op": stage1.skeleton_graph_op,
            },
        )
        temporal_block_type = infer_temporal_block_type_from_checkpoint(handle.name)
    assert temporal_block_type == "attention"


def test_load_checkpoint_rejects_legacy_32_joint_layout_metadata():
    stage1 = Stage1Model(latent_dim=32, skeleton_graph_op="gcn")
    bad_payload = {
        "state_dict": stage1.state_dict(),
        "extra": {
            "skeleton_layout_version": "legacy_32j",
            "num_joints": 32,
            "encoder_graph_op": stage1.encoder_graph_op,
            "skeleton_graph_op": stage1.skeleton_graph_op,
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(bad_payload, handle.name)
        with pytest.raises(ValueError, match="incompatible skeleton layout"):
            load_checkpoint(handle.name, stage1, strict=False)


def test_checkpoint_graph_inference_rejects_missing_layout_metadata():
    stage1 = Stage1Model(latent_dim=32, skeleton_graph_op="gat", encoder_type="gcn")
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save({"state_dict": stage1.state_dict()}, handle.name)
        with pytest.raises(ValueError, match="incompatible skeleton layout"):
            infer_graph_ops_from_checkpoint(handle.name)


def test_imu_graph_adjacency_shape_and_edges():
    """build_imu_graph_adjacency must produce a [2T,2T] adjacency with the
    correct self-loops, within-stream temporal edges, and cross-sensor edges."""
    T = 4
    adj = build_imu_graph_adjacency(window_len=T, device=torch.device("cpu"))

    assert adj.shape == (2 * T, 2 * T), f"Expected [{2*T},{2*T}], got {adj.shape}"
    # Self-loops
    for i in range(2 * T):
        assert adj[i, i] == 1.0, f"Missing self-loop at node {i}"
    # Hip intra-stream temporal edges
    for i in range(T - 1):
        assert adj[i, i + 1] == 1.0 and adj[i + 1, i] == 1.0, f"Missing hip temporal edge ({i},{i+1})"
    # Wrist intra-stream temporal edges
    for i in range(T - 1):
        assert adj[T + i, T + i + 1] == 1.0 and adj[T + i + 1, T + i] == 1.0, f"Missing wrist temporal edge"
    # Cross-sensor edges
    for i in range(T):
        assert adj[i, T + i] == 1.0 and adj[T + i, i] == 1.0, f"Missing cross-sensor edge at t={i}"
    # No edges between non-adjacent hip nodes (only direct neighbours)
    assert adj[0, 2] == 0.0, "Unexpected non-adjacent hip edge"


def test_imu_graph_encoder_output_shapes():
    """IMUGraphEncoder must return per-sensor tokens of the expected shape."""
    B, T, D = 2, 8, 32
    encoder = IMUGraphEncoder(input_dim=len(IMU_FEATURE_NAMES), latent_dim=D, depth=2)
    hip = torch.randn(B, T, len(IMU_FEATURE_NAMES))
    wrist = torch.randn(B, T, len(IMU_FEATURE_NAMES))
    hip_out, wrist_out = encoder(hip, wrist)
    assert hip_out.shape == (B, T, D), f"hip_out shape mismatch: {hip_out.shape}"
    assert wrist_out.shape == (B, T, D), f"wrist_out shape mismatch: {wrist_out.shape}"
    assert torch.isfinite(hip_out).all()
    assert torch.isfinite(wrist_out).all()


def test_imu_aligner_uses_dual_sensor_encoders():
    """IMULatentAligner must expose the current dual-encoder contract and produce correct output shapes."""
    B, T, D = 2, 8, 32
    aligner = IMULatentAligner(latent_dim=D)
    assert hasattr(aligner, "hip_encoder")
    assert hasattr(aligner, "wrist_encoder")
    assert aligner.imu_feature_dim == len(IMU_FEATURE_NAMES)
    a_hip = torch.randn(B, T, 3)
    a_wrist = torch.randn(B, T, 3)
    sensor_tokens, h_global = aligner(a_hip, a_wrist)
    assert sensor_tokens.shape == (B, T, D)
    assert h_global.shape == (B, D)
    assert torch.isfinite(sensor_tokens).all()
    assert torch.isfinite(h_global).all()


def test_stage3_augment_conditioning_is_noop_without_or_before_shared_signal():
    x, a_hip, a_wrist, gait, _ = _dummy_batch(batch_size=1)
    stage1 = Stage1Model(
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=32, num_joints=DEFAULT_JOINTS, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=32,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    h_tokens, h_global = stage2.aligner(a_hip, a_wrist)
    same_tokens, same_global = stage3.augment_conditioning(h_tokens, h_global)
    aug_tokens, aug_global = stage3.augment_conditioning(h_tokens, h_global, a_hip_stream=a_hip, a_wrist_stream=a_wrist)

    assert torch.allclose(same_tokens, h_tokens)
    assert torch.allclose(same_global, h_global)
    assert aug_tokens.shape == h_tokens.shape
    assert aug_global.shape == h_global.shape
    assert torch.allclose(aug_tokens, h_tokens)
    assert torch.allclose(aug_global, h_global)


def test_rotate_and_align_torch_handles_degenerate_sequences():
    pose = torch.zeros(90, DEFAULT_JOINTS, 3)
    aligned = rotate_and_align_torch(pose)
    assert aligned.shape == (90, 16, 3)
    assert torch.isfinite(aligned).all()


def test_write_classification_artifacts_outputs_expected_files():
    y_true = np.array([0, 1, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 2], dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        report = write_classification_artifacts(out_dir, "demo_classifier", y_true, y_pred, num_classes=3)
        assert report["accuracy"] == pytest.approx(0.75)
        assert (out_dir / "demo_classifier_report.json").exists()
        assert (out_dir / "demo_classifier_report.csv").exists()
        assert (out_dir / "demo_classifier_confusion_matrix.csv").exists()
        assert (out_dir / "demo_classifier_confusion_matrix.png").exists()


def test_write_history_emits_accuracy_curves_when_accuracy_columns_exist():
    history = [
        {
            "epoch": 1.0,
            "train_loss_total": 1.0,
            "val_loss_total": 1.2,
            "train_acc_latent_cls": 0.5,
            "val_acc_latent_cls": 0.4,
        },
        {
            "epoch": 2.0,
            "train_loss_total": 0.8,
            "val_loss_total": 1.0,
            "train_acc_latent_cls": 0.6,
            "val_acc_latent_cls": 0.5,
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        write_history(run_dir, "stage1", history)
        assert (run_dir / "stage1" / "history.csv").exists()
        assert (run_dir / "stage1" / "accuracy_curves.png").exists()


def test_stage1_evaluator_writes_latent_classifier_artifacts():
    loader = DataLoader(_EvalDummyDataset(length=4, frames=8), batch_size=2, shuffle=False)
    stage1 = Stage1Model(
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        evaluate_stage1(stage1, loader, torch.device("cpu"), out_dir, timestep_values=[0, 9])
        assert (out_dir / "latent_classifier_report.json").exists()
        assert (out_dir / "latent_classifier_confusion_matrix.csv").exists()


def test_stage2_lightweight_evaluator_writes_classifier_and_gait_reports():
    loader = DataLoader(_EvalDummyDataset(length=4, frames=8), batch_size=2, shuffle=False)
    stage1 = Stage1Model(
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        evaluate_stage2_reports(stage2, loader, torch.device("cpu"), out_dir, max_batches=2)
        assert (out_dir / "imu_classifier_report.json").exists()
        assert (out_dir / "imu_classifier_confusion_matrix.csv").exists()
        assert (out_dir / "gait_prediction_metrics.json").exists()
        assert (out_dir / "gait_prediction_metrics.csv").exists()


def test_stage2_cli_defaults_preserve_heavy_eval_and_add_light_report_interval(monkeypatch):
    import train as train_module

    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", "1"])
    args = train_module.parse_args()
    assert args.eval_every_stage2 == 500
    assert args.eval_every_stage2_reports == 10


def test_stage2_cli_defaults_include_regularization_knobs():
    import train as train_module

    args = train_module.parse_args(["--stage", "2"])
    assert args.stage2_dropout == pytest.approx(0.25)
    assert args.stage2_weight_decay == pytest.approx(3e-3)
    assert args.stage2_aug_noise_std == pytest.approx(0.01)
    assert args.stage2_aug_scale == pytest.approx(0.05)
    assert args.stage2_aug_mask_prob == pytest.approx(0.05)
    assert args.stage2_scheduler_patience == 3
    assert args.stage2_scheduler_factor == pytest.approx(0.5)
    assert args.stage2_scheduler_min_lr == pytest.approx(1e-6)


def test_stage2_optimizer_and_scheduler_use_stage2_specific_defaults():
    import train as train_module

    args = train_module.parse_args(["--stage", "2"])
    stage1 = Stage1Model(latent_dim=16, num_joints=DEFAULT_JOINTS, timesteps=10, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        stage2_dropout=args.stage2_dropout,
    )
    optimizer, scheduler = train_module._build_stage2_optimizer_and_scheduler(stage2, args)

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(3e-3)
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 3
    assert scheduler.cooldown == 1
    assert scheduler.threshold == pytest.approx(1e-3)
    assert scheduler.min_lrs == [pytest.approx(1e-6)]


def test_stage2_sensor_encoder_dropout_defaults_and_overrides():
    encoder = SensorGCNEncoder(input_dim=len(IMU_FEATURE_NAMES), latent_dim=16)
    assert encoder.drop1.p == pytest.approx(0.25)
    assert encoder.drop2.p == pytest.approx(0.25)
    assert encoder.drop3.p == pytest.approx(0.25)

    aligner = IMULatentAligner(latent_dim=16, dropout=0.4)
    assert aligner.hip_encoder.drop1.p == pytest.approx(0.4)
    assert aligner.wrist_encoder.drop3.p == pytest.approx(0.4)


def test_stage2_augmentation_changes_train_streams_but_not_eval_streams():
    import train as train_module

    a_hip = torch.randn(2, 8, 3)
    a_wrist = torch.randn(2, 8, 3)
    args = SimpleNamespace(
        stage2_aug_noise_std=0.01,
        stage2_aug_scale=0.05,
        stage2_aug_mask_prob=0.05,
    )

    torch.manual_seed(0)
    aug_hip, aug_wrist = train_module._maybe_augment_stage2_streams(a_hip, a_wrist, args, training=True)
    same_hip, same_wrist = train_module._maybe_augment_stage2_streams(a_hip, a_wrist, args, training=False)

    assert torch.allclose(same_hip, a_hip)
    assert torch.allclose(same_wrist, a_wrist)
    assert not torch.allclose(aug_hip, a_hip)
    assert not torch.allclose(aug_wrist, a_wrist)


def test_stage2_checkpoint_metadata_saves_sensor_locations():
    import train as train_module

    args = train_module.parse_args(
        ["--stage", "2", "--hip_folder", "/tmp/meta_hip", "--wrist_folder", "/tmp/meta_wrist"]
    )
    args.encoder_graph_op_resolved = "gcn"
    args.skeleton_graph_op_resolved = "gcn"
    args.run_dir = "outputs/test_stage2"
    metadata = train_module._stage2_checkpoint_metadata(args)

    stage1 = Stage1Model(latent_dim=16, num_joints=DEFAULT_JOINTS, timesteps=10, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    stage2 = Stage2Model(encoder=stage1.encoder, latent_dim=16, num_joints=DEFAULT_JOINTS, gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        save_checkpoint(handle.name, stage2, extra=metadata)
        payload = torch.load(handle.name, map_location="cpu")

    assert payload["extra"]["sensor_locations"] == ["meta_hip", "meta_wrist"]


def test_stage3_warns_on_stage2_sensor_domain_mismatch(caplog):
    import train as train_module

    args = train_module.parse_args(
        ["--stage", "3", "--hip_folder", "/tmp/phone", "--wrist_folder", "/tmp/watch"]
    )
    checkpoint = {"extra": {"sensor_locations": ["meta_hip", "meta_wrist"]}}

    with caplog.at_level(logging.WARNING, logger="train"):
        warned = train_module._warn_stage2_stage3_sensor_domain_mismatch(checkpoint, args)

    assert warned is True
    assert "Stage2 sensor_locations=['meta_hip', 'meta_wrist']" in caplog.text
    assert "Stage3 sensor_locations=['phone', 'watch']" in caplog.text


def test_stage3_evaluator_writes_eval_history_and_classifier_reports():
    loader = DataLoader(_EvalDummyDataset(length=4, frames=8), batch_size=2, shuffle=False)
    stage1 = Stage1Model(
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
    )
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "epoch_001"
        evaluate_stage3(
            stage2,
            stage3,
            loader,
            torch.device("cpu"),
            out_dir,
            sample_steps=2,
            fps=30.0,
            epoch=1,
            sampler="ddim",
            sample_seed=0,
        )
        assert (out_dir / "real_classifier_report.json").exists()
        assert (out_dir / "generated_classifier_report.json").exists()
        assert (out_dir.parent / "eval_history.csv").exists()
        assert (out_dir.parent / "eval_accuracy_curves.png").exists()
        assert (out_dir / "per_class_accuracy.json").exists()


def test_stage3_per_class_accuracy_uses_raw_real_skeletons_for_real_accuracy():
    loader = DataLoader(_EvalDummyDataset(length=4, frames=8), batch_size=2, shuffle=False)
    stage1 = Stage1Model(
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
    )
    stage2 = Stage2Model(
        encoder=stage1.encoder,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
    )
    stage3 = Stage3Model(
        encoder=stage1.encoder,
        decoder=stage1.decoder,
        denoiser=stage1.denoiser,
        latent_normalizer=stage1.latent_normalizer,
        latent_dim=16,
        num_joints=DEFAULT_JOINTS,
        num_classes=14,
        timesteps=10,
        gait_metrics_dim=DEFAULT_GAIT_METRICS_DIM,
        use_gait_conditioning=False,
        shared_motion_layer=stage2.shared_motion_layer,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = plot_per_class_accuracy(
            stage2,
            stage3,
            loader,
            torch.device("cpu"),
            Path(tmpdir),
            sample_steps=2,
            sampler="ddim",
            max_batches=2,
        )
        manual_correct = 0
        manual_total = 0
        for batch in loader:
            x = batch["skeleton"]
            y = batch["label"]
            pred_real = stage3.classifier(x.float()).argmax(1)
            manual_correct += int((pred_real == y).sum().item())
            manual_total += int(y.numel())
        assert result["overall_real_acc"] == pytest.approx(manual_correct / max(manual_total, 1))


# ── LatentNormalizer tests ──────────────────────────────────────────


def test_latent_normalizer_normalize_denormalize_roundtrip():
    from diffusion_model.model import LatentNormalizer
    norm = LatentNormalizer(latent_dim=32)
    norm.train()
    z = torch.randn(2, 8, DEFAULT_JOINTS, 32) * 5 + 3  # mean=3, std=5
    z_n = norm.normalize(z)
    z_back = norm.denormalize(z_n)
    assert torch.allclose(z, z_back, atol=1e-4)
    assert abs(z_n.mean().item()) < 0.5
    assert abs(z_n.std().item() - 1.0) < 0.5
    # After eval, stats frozen — different input still round-trips
    norm.eval()
    z2 = torch.randn(2, 8, DEFAULT_JOINTS, 32) * 10
    z2_back = norm.denormalize(norm.normalize(z2))
    assert torch.allclose(z2, z2_back, atol=1e-4)


def test_latent_normalizer_buffers_in_state_dict():
    from diffusion_model.model import LatentNormalizer
    norm = LatentNormalizer(latent_dim=16)
    norm.train()
    z = torch.randn(2, 4, 8, 16) * 3
    norm.normalize(z)  # trigger stats update
    sd = norm.state_dict()
    assert "running_mean" in sd
    assert "running_var" in sd
    assert "num_batches_tracked" in sd
    assert sd["num_batches_tracked"].item() == 1
    # Load into fresh normalizer
    norm2 = LatentNormalizer(latent_dim=16)
    norm2.load_state_dict(sd)
    assert torch.allclose(norm.running_mean, norm2.running_mean)
    assert torch.allclose(norm.running_var, norm2.running_var)
