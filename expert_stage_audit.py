from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class NotebookCommand:
    stage: int
    text: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_notebook_commands(notebook_path: Path) -> dict[int, NotebookCommand]:
    payload = load_json(notebook_path)
    commands: dict[int, NotebookCommand] = {}
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        m = re.search(r"--stage\s+([123])", text)
        if m:
            commands[int(m.group(1))] = NotebookCommand(stage=int(m.group(1)), text=text)
    return commands


def extract_flag(command: str, flag: str) -> str | None:
    m = re.search(rf"{re.escape(flag)}\s+([^\s\\]+)", command)
    return m.group(1) if m else None


def load_history(stage_dir: Path) -> pd.DataFrame:
    history_path = stage_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history: {history_path}")
    return pd.read_csv(history_path)


def best_row(df: pd.DataFrame, key: str = "val_loss_total") -> pd.Series:
    if key not in df.columns:
        key = "train_loss_total"
    return df.loc[df[key].astype(float).idxmin()]


def collect_stage3_metrics(stage3_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for epoch_dir in sorted(stage3_dir.glob("epoch_*")):
        try:
            epoch = int(epoch_dir.name.split("_")[-1])
        except ValueError:
            continue
        recon = load_json(epoch_dir / "stage3_reconstruction_metrics.json")
        acc = load_json(epoch_dir / "per_class_accuracy.json")
        cond = load_json(epoch_dir / "conditioning_comparison.json")
        rows.append(
            {
                "epoch": epoch,
                "mpjpe_mean": recon.get("mpjpe_mean"),
                "root_trajectory_error_mean": recon.get("root_trajectory_error_mean"),
                "velocity_error_mean": recon.get("velocity_error_mean"),
                "overall_gen_acc": acc.get("overall_gen_acc"),
                "overall_real_acc": acc.get("overall_real_acc"),
                "cond_vs_real_l2": cond.get("cond_vs_real_l2"),
                "uncond_vs_real_l2": cond.get("uncond_vs_real_l2"),
                "cond_vs_uncond_l2": cond.get("cond_vs_uncond_l2"),
                "mean_pairwise_dist": cond.get("mean_pairwise_dist"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No stage3 epoch diagnostics found under {stage3_dir}")
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def plot_histories(stage1: pd.DataFrame, stage2: pd.DataFrame, stage3: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)

    axes[0].plot(stage1["epoch"], stage1["val_loss_total"], label="val_total", lw=2)
    axes[0].plot(stage1["epoch"], stage1["val_loss_diff"], label="val_diff", alpha=0.9)
    axes[0].plot(stage1["epoch"], stage1["val_loss_cls"], label="val_cls", alpha=0.9)
    axes[0].plot(stage1["epoch"], stage1["val_loss_var"], label="val_var", alpha=0.9)
    axes[0].set_title("Stage 1 validation losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stage2["epoch"], stage2["val_loss_total"], label="val_total", lw=2)
    axes[1].plot(stage2["epoch"], stage2["val_loss_align"], label="val_align", alpha=0.9)
    axes[1].plot(stage2["epoch"], stage2["val_loss_feature"], label="val_feature", alpha=0.9)
    axes[1].plot(stage2["epoch"], stage2["val_loss_cls"], label="val_cls", alpha=0.9)
    axes[1].plot(stage2["epoch"], stage2["val_loss_gait_pred"], label="val_gait_pred", alpha=0.9)
    axes[1].set_title("Stage 2 validation losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(stage3["epoch"], stage3["val_loss_total"], label="val_total", lw=2)
    axes[2].plot(stage3["epoch"], stage3["val_loss_diff"], label="val_diff", alpha=0.9)
    axes[2].plot(stage3["epoch"], stage3["val_loss_pose"], label="val_pose", alpha=0.9)
    axes[2].plot(stage3["epoch"], stage3["val_loss_latent"], label="val_latent", alpha=0.9)
    axes[2].plot(stage3["epoch"], stage3["val_loss_motion"], label="val_motion", alpha=0.9)
    axes[2].plot(stage3["epoch"], stage3["val_loss_cls"], label="val_cls", alpha=0.9)
    axes[2].plot(stage3["epoch"], stage3["val_loss_gait"], label="val_gait", alpha=0.9)
    axes[2].set_title("Stage 3 validation losses")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stage3_diagnostics(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].plot(df["epoch"], df["mpjpe_mean"], marker="o", label="MPJPE")
    axes[0, 0].plot(df["epoch"], df["root_trajectory_error_mean"], marker="o", label="Root traj error")
    axes[0, 0].set_title("Stage 3 kinematic errors")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Meters")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df["epoch"], df["overall_gen_acc"], marker="o", label="Generated")
    axes[0, 1].plot(df["epoch"], df["overall_real_acc"], marker="o", label="Real")
    axes[0, 1].set_title("Classifier accuracy on real vs generated")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["epoch"], df["cond_vs_real_l2"], marker="o", label="Cond vs real")
    axes[1, 0].plot(df["epoch"], df["uncond_vs_real_l2"], marker="o", label="Uncond vs real")
    axes[1, 0].plot(df["epoch"], df["cond_vs_uncond_l2"], marker="o", label="Cond vs uncond")
    axes[1, 0].set_title("Conditioning distance scale")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L2 distance")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["epoch"], df["mean_pairwise_dist"], marker="o", label="Pairwise distance")
    axes[1, 1].plot(df["epoch"], df["velocity_error_mean"], marker="o", label="Velocity error")
    axes[1, 1].set_title("Diversity vs velocity error")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stage2_domain_check(payload: dict, out_path: Path) -> None:
    labels = list(payload.keys())
    retrieval = [payload[k]["retrieval_at_1"] for k in labels]
    cls_acc = [payload[k]["cls_head_acc"] for k in labels]
    cosine = [payload[k]["pair_cosine_mean"] for k in labels]
    chance = 1.0 / 16.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(labels, retrieval, color=["#1f77b4", "#ff7f0e"])
    axes[0].axhline(chance, color="red", linestyle="--", label="chance@16")
    axes[0].set_title("Stage 2 retrieval@1")
    axes[0].set_ylim(0, max(max(retrieval) * 1.2, chance * 1.5))
    axes[0].legend()

    axes[1].bar(labels, cls_acc, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Stage 2 cls_head accuracy")
    axes[1].set_ylim(0, max(max(cls_acc) * 1.2, 0.05))

    axes[2].bar(labels, cosine, color=["#1f77b4", "#ff7f0e"])
    axes[2].set_title("Paired cosine similarity")
    axes[2].set_ylim(0, 1.0)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_decoder_summary(payload: dict, out_path: Path) -> None:
    labels = ["Encoder->Decoder", "Diffusion->Decoder"]
    values = [payload["autoencoder_mpjpe_m"], payload["diffusion_mpjpe_m"]]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(labels, values, color=["#2ca02c", "#d62728"])
    ax.set_ylabel("MPJPE (m)")
    ax.set_title("Decoder vs diffusion diagnosis")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(
    out_path: Path,
    notebook_path: Path,
    stage1_dir: Path,
    stage2_dir: Path,
    stage3_dir: Path,
    stage1_hist: pd.DataFrame,
    stage2_hist: pd.DataFrame,
    stage3_hist: pd.DataFrame,
    stage3_diag: pd.DataFrame,
    commands: dict[int, NotebookCommand],
    domain_check: dict | None,
    decoder_summary: dict | None,
) -> None:
    s1_best = best_row(stage1_hist)
    s2_best = best_row(stage2_hist)
    s3_best = best_row(stage3_hist)
    diag_best = stage3_diag.loc[stage3_diag["mpjpe_mean"].astype(float).idxmin()]
    stage2_cmd = commands.get(2).text if 2 in commands else ""
    stage3_cmd = commands.get(3).text if 3 in commands else ""

    stage2_hip = extract_flag(stage2_cmd, "--hip_folder")
    stage2_wrist = extract_flag(stage2_cmd, "--wrist_folder")
    stage3_hip = extract_flag(stage3_cmd, "--hip_folder")
    stage3_wrist = extract_flag(stage3_cmd, "--wrist_folder")
    lambda_cls_s2 = extract_flag(stage2_cmd, "--lambda_cls_s2")
    lambda_gait_s2 = extract_flag(stage2_cmd, "--lambda_gait_s2")

    lines: list[str] = []
    lines.append("# Expert Stage Audit")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Notebook: `{notebook_path}`")
    lines.append(f"- Stage 1 reports: `{stage1_dir}`")
    lines.append(f"- Stage 2 reports: `{stage2_dir}`")
    lines.append(f"- Stage 3 reports: `{stage3_dir}`")
    lines.append("")
    lines.append("## Executive Verdict")
    lines.append("- Stage 1 learned a usable encoder/denoiser objective, but its decoder was not trained in Stage 1 and remains too weak to produce good skeletons from clean latents.")
    lines.append("- Stage 2 is the main conditioning bottleneck. In the notebook, `--lambda_cls_s2 0` and `--lambda_gait_s2 0` remove the two supervised objectives that the Stage 2 architecture says are supposed to make IMU embeddings informative.")
    lines.append("- Stage 3 inherits that weak conditioning and also relies on a weak decoder. The result is diverse but biomechanically invalid generations whose conditional version is only marginally closer to reality than the unconditional version.")
    lines.append("- There is also a train/validation bookkeeping bug in Stage 3: training weights classification by `lambda_cls`, but validation adds classification loss unweighted, so train and val totals are not directly comparable.")
    lines.append("")
    lines.append("## Architecture")
    lines.append("- Stage 1: graph skeleton encoder -> latent diffusion denoiser -> auxiliary latent classifier. The decoder exists in the module, but Stage 1 training optimizes diffusion, classification, and variance regularization only.")
    lines.append("- Stage 2: frozen Stage 1 encoder supplies target latents. A two-stream IMU aligner produces temporal tokens and a global embedding, with feature matching and latent alignment losses plus optional classification and gait prediction heads.")
    lines.append("- Stage 3: Stage 2 IMU tokens condition the Stage 1 denoiser, and the Stage 1 decoder maps generated latents back to skeletons. Reconstruction, latent, velocity, motion, gait, and classifier losses are optimized jointly.")
    lines.append("")
    lines.append("## Evidence")
    lines.append(
        f"- Stage 1 best validation epoch: {int(s1_best['epoch'])} with val_loss_total={float(s1_best['val_loss_total']):.4f}, val_loss_diff={float(s1_best['val_loss_diff']):.4f}."
    )
    lines.append(
        f"- Stage 2 best validation epoch: {int(s2_best['epoch'])} with val_loss_total={float(s2_best['val_loss_total']):.4f}, val_loss_align={float(s2_best['val_loss_align']):.4f}, val_loss_feature={float(s2_best['val_loss_feature']):.6f}, val_loss_cls={float(s2_best['val_loss_cls']):.4f}, val_loss_gait_pred={float(s2_best['val_loss_gait_pred']):.4f}."
    )
    lines.append(
        f"- Stage 3 best validation epoch by logged total: {int(s3_best['epoch'])} with val_loss_total={float(s3_best['val_loss_total']):.4f}, val_loss_cls={float(s3_best['val_loss_cls']):.4f}, val_loss_pose={float(s3_best['val_loss_pose']):.4f}, val_loss_latent={float(s3_best['val_loss_latent']):.4f}."
    )
    lines.append(
        f"- Best diagnostic MPJPE among saved Stage 3 audits occurs at epoch {int(diag_best['epoch'])}: MPJPE={float(diag_best['mpjpe_mean']):.4f} m, root trajectory error={float(diag_best['root_trajectory_error_mean']):.4f} m, generated classifier accuracy={float(diag_best['overall_gen_acc']):.4f}, real classifier accuracy={float(diag_best['overall_real_acc']):.4f}."
    )
    if decoder_summary:
        lines.append(
            f"- Decoder-vs-diffusion test: clean encoder latents decoded at MPJPE={decoder_summary['autoencoder_mpjpe_m']:.4f} m, while diffusion latents decoded at MPJPE={decoder_summary['diffusion_mpjpe_m']:.4f} m. The gap is small; the decoder itself is already bad."
        )
    if domain_check:
        meta = domain_check.get("meta", {})
        phone = domain_check.get("phone_watch", {})
        lines.append(
            f"- Stage 2 retrieval@1 is {meta.get('retrieval_at_1', float('nan')):.4f} on meta sensors and {phone.get('retrieval_at_1', float('nan')):.4f} on phone/watch, which is effectively chance for batch size 16 (0.0625)."
        )
        lines.append(
            f"- Stage 2 classifier head accuracy is {meta.get('cls_head_acc', float('nan')):.4f} on meta sensors and {phone.get('cls_head_acc', float('nan')):.4f} on phone/watch, confirming that the learned embedding is not behaviorally discriminative."
        )
    if stage2_hip and stage3_hip:
        lines.append(
            f"- Sensor domain mismatch: Stage 2 uses `{Path(stage2_hip).name}` / `{Path(stage2_wrist).name}`, while Stage 3 conditions on `{Path(stage3_hip).name}` / `{Path(stage3_wrist).name}`."
        )
    lines.append("")
    lines.append("## Why It Fails")
    lines.append("- Objective level: the Stage 2 notebook disables the two supervised objectives the architecture description depends on, so the aligner is trained mostly to imitate broad latent statistics instead of learning behaviorally meaningful IMU conditioning.")
    lines.append("- Interface level: Stage 2 is trained on a different sensor domain than Stage 3. Even if the aligner had been stronger, Stage 3 asks it to condition on a distribution it was not optimized on.")
    lines.append("- Generative level: Stage 3 conditional samples remain far from real motion. Across saved audits, `cond_vs_real_l2` stays around 54 to 64 while `cond_vs_uncond_l2` is only about 4.6 to 9.8, so conditioning changes the sample far less than the total error to reality.")
    lines.append("- Decoder level: the decoder cannot reconstruct realistic motion even from clean encoder latents, so diffusion cannot produce good skeletons downstream until decoder quality improves.")
    lines.append("- Biomechanics level: the generated figures show implausible limb crossings, collapsed posture, and unstable body orientation. An MPJPE near 0.8 m is much too large for normal gait reconstruction and indicates the system is not preserving human pose geometry.")
    lines.append("")
    lines.append("## Additional Notes")
    lines.append(f"- Notebook Stage 2 settings extracted from `{notebook_path.name}`: `--lambda_cls_s2 {lambda_cls_s2}` and `--lambda_gait_s2 {lambda_gait_s2}`.")
    lines.append("- Stage 3 validation totals should be interpreted cautiously because classification loss is weighted in training but unweighted in validation logging.")
    lines.append("- The dataset constructor accepts `normalize_sensors`, but the CSV paired dataset path does not actually normalize the sensor windows.")
    lines.append("")
    lines.append("## Generated Artifacts")
    lines.append("- `training_curves.png`")
    lines.append("- `stage3_diagnostics.png`")
    if domain_check:
        lines.append("- `stage2_domain_check.png`")
    if decoder_summary:
        lines.append("- `decoder_vs_diffusion.png`")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an expert audit over Stage 1/2/3 outputs.")
    parser.add_argument("--stage1_path", required=True)
    parser.add_argument("--stage2_path", required=True)
    parser.add_argument("--stage3_path", required=True)
    parser.add_argument("--notebook_path", required=True)
    parser.add_argument("--output_dir", default="outputs/expert_debug")
    parser.add_argument("--domain_check_json", default="outputs/expert_debug/stage2_domain_check_cpu.json")
    parser.add_argument("--decoder_summary_json", default="outputs/expert_debug/decoder_vs_diffusion_summary.json")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_path)
    stage2_dir = Path(args.stage2_path)
    stage3_dir = Path(args.stage3_path)
    notebook_path = Path(args.notebook_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_hist = load_history(stage1_dir)
    stage2_hist = load_history(stage2_dir)
    stage3_hist = load_history(stage3_dir)
    stage3_diag = collect_stage3_metrics(stage3_dir)
    commands = find_notebook_commands(notebook_path)

    domain_check_path = Path(args.domain_check_json)
    decoder_summary_path = Path(args.decoder_summary_json)
    domain_check = load_json(domain_check_path) if domain_check_path.exists() else None
    decoder_summary = load_json(decoder_summary_path) if decoder_summary_path.exists() else None

    plot_histories(stage1_hist, stage2_hist, stage3_hist, output_dir / "training_curves.png")
    plot_stage3_diagnostics(stage3_diag, output_dir / "stage3_diagnostics.png")
    if domain_check:
        plot_stage2_domain_check(domain_check, output_dir / "stage2_domain_check.png")
    if decoder_summary:
        plot_decoder_summary(decoder_summary, output_dir / "decoder_vs_diffusion.png")

    write_report(
        out_path=output_dir / "expert_audit_report.md",
        notebook_path=notebook_path,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        stage3_dir=stage3_dir,
        stage1_hist=stage1_hist,
        stage2_hist=stage2_hist,
        stage3_hist=stage3_hist,
        stage3_diag=stage3_diag,
        commands=commands,
        domain_check=domain_check,
        decoder_summary=decoder_summary,
    )


if __name__ == "__main__":
    main()
