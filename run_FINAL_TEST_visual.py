from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_cluster_I8_gated_visual_residual as S
from dft321_validation.dft321 import dft321


# ============================================================
# FINAL FROZEN TEST
# ============================================================

DEVICE = "cuda:0"
BATCH_SIZE = 512
NUM_WORKERS = 0
AMP_ENABLED = True

FS = 8000.0
HISTORY_N = 1280
FUTURE_N = 160

EXPECTED_TEST_TRIALS = 9440
EXPECTED_TEST_WINDOWS = 28320
EXPECTED_METRIC_ROWS = 84960

EXPECTED_INDEX_SHA256 = (
    "7780a63a2d626c0855f97109de4152c"
    "109800500f3d4c1a77132ddee0d72aec8"
)

INDEX_PATH = Path(
    "cluster_FINAL_TEST_results/"
    "final_test_eval_index.csv"
)

CHECKPOINT = Path(
    "cluster_I8_gated_visual_residual_results/"
    "logmag/lambda_0.01/"
    "seed_42_full/best_frequency.pt"
)

OUT = Path(
    "cluster_FINAL_TEST_visual_results"
)

FIG_DIR = OUT / "figures"
PRED_DIR = OUT / "predictions"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def robust_median(x):
    x = np.asarray(
        x,
        dtype=np.float64
    )
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.median(x)
    )


def build_model(ckpt, device):

    args = ckpt["args"]
    protocol = ckpt["protocol"]

    required_protocol = {
        "history_n": 1280,
        "future_n": 160,
        "stride_n": 160,
        "test_accessed": False,
    }

    for k, expected in (
        required_protocol.items()
    ):
        got = protocol.get(k)

        if got != expected:
            raise RuntimeError(
                f"Checkpoint protocol mismatch: "
                f"{k}={got}, expected={expected}"
            )

    if not np.isclose(
        float(protocol["fs"]),
        8000.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Checkpoint fs != 8000"
        )

    if args.get("fusion") != "multifilm":
        raise RuntimeError(
            "Final checkpoint fusion != multifilm"
        )

    if args.get("condition") != "full":
        raise RuntimeError(
            "Final checkpoint condition != full"
        )

    if not np.isclose(
        float(args.get("lambda_spec")),
        0.01,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Final checkpoint lambda_spec != 0.01"
        )

    model = S.ConditionedTransformer(
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        dropout=float(args["dropout"]),
        patch_len=20,
        nhead=4,
        fusion="multifilm",
        condition_dim=4,
    )

    model.load_state_dict(
        ckpt["model_state"],
        strict=True
    )

    model = model.to(device)
    model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    if params != 471617:
        raise RuntimeError(
            f"Parameter mismatch: {params}"
        )

    return model, params


def compute_dynamic_strength(gt):
    """
    gt: [160,3]

    GT-only dynamic strength:
    RMS of first differences across
    time and XYZ.
    """
    d = np.diff(
        np.asarray(
            gt,
            dtype=np.float64
        ),
        axis=0
    )

    return float(
        np.sqrt(
            np.mean(
                d ** 2
            )
        )
    )


def select_visual_windows(
    meta_df,
    gt_all,
):
    """
    Selection is strictly GT-only.

    No prediction values or prediction
    metrics enter this function.
    """

    strengths = np.asarray(
        [
            compute_dynamic_strength(gt)
            for gt in gt_all
        ],
        dtype=np.float64
    )

    if not np.isfinite(
        strengths
    ).all():
        raise RuntimeError(
            "Non-finite GT dynamic strength"
        )

    median_target = float(
        np.median(strengths)
    )

    p95_target = float(
        np.percentile(
            strengths,
            95
        )
    )

    tmp = meta_df.copy()

    tmp["gt_dynamic_strength"] = (
        strengths
    )

    # Deterministic tie-break.
    tmp = tmp.sort_values(
        "window_key_sha256",
        kind="stable"
    )

    typical_idx = (
        tmp.assign(
            distance=(
                tmp["gt_dynamic_strength"]
                -
                median_target
            ).abs()
        )
        .sort_values(
            [
                "distance",
                "window_key_sha256",
            ],
            kind="stable"
        )
        .index[0]
    )

    high_idx = (
        tmp.assign(
            distance=(
                tmp["gt_dynamic_strength"]
                -
                p95_target
            ).abs()
        )
        .sort_values(
            [
                "distance",
                "window_key_sha256",
            ],
            kind="stable"
        )
        .index[0]
    )

    strongest_idx = (
        tmp.sort_values(
            [
                "gt_dynamic_strength",
                "window_key_sha256",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable"
        )
        .index[0]
    )

    selected = {
        "typical":
            int(typical_idx),

        "high_dynamic":
            int(high_idx),

        "strongest_transient":
            int(strongest_idx),
    }

    return (
        selected,
        strengths,
        median_target,
        p95_target,
    )


def plot_xyz(
    name,
    gt,
    pred,
    meta,
    out_path,
):
    t_ms = (
        np.arange(FUTURE_N)
        /
        FS
        *
        1000.0
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 6.8),
        sharex=True,
    )

    labels = ["X", "Y", "Z"]

    for ai, ax in enumerate(axes):

        ax.plot(
            t_ms,
            gt[:, ai],
            linewidth=1.2,
            label="Ground Truth",
        )

        ax.plot(
            t_ms,
            pred[:, ai],
            linewidth=1.2,
            linestyle="--",
            label="Prediction",
        )

        ax.set_ylabel(
            f"{labels[ai]} acceleration"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        if ai == 0:
            ax.legend(
                loc="best",
                frameon=False
            )

    axes[-1].set_xlabel(
        "Forecast time (ms)"
    )

    fig.suptitle(
        (
            f"{name} | "
            f"Texture {int(meta['texture_id'])}, "
            f"{int(meta['velocity'])} mm/s, "
            f"{int(meta['direction'])}°, "
            f"{int(meta['force_mN'])} mN"
        ),
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_dft321(
    name,
    gt,
    pred,
    out_path,
):
    gt321 = np.asarray(
        dft321(gt),
        dtype=np.float64
    ).reshape(-1)

    pr321 = np.asarray(
        dft321(pred),
        dtype=np.float64
    ).reshape(-1)

    if (
        gt321.shape != (FUTURE_N,)
        or
        pr321.shape != (FUTURE_N,)
    ):
        raise RuntimeError(
            "DFT321 output shape mismatch"
        )

    t_ms = (
        np.arange(FUTURE_N)
        /
        FS
        *
        1000.0
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 3.2)
    )

    ax.plot(
        t_ms,
        gt321,
        linewidth=1.2,
        label="Ground Truth",
    )

    ax.plot(
        t_ms,
        pr321,
        linewidth=1.2,
        linestyle="--",
        label="Prediction",
    )

    ax.set_xlabel(
        "Forecast time (ms)"
    )

    ax.set_ylabel(
        "DFT321 acceleration"
    )

    ax.set_title(
        name
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return gt321, pr321


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "FINAL FROZEN REPEAT-1 TEST"
    )
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        DEVICE
    )

    torch.cuda.set_device(
        device
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Frozen index integrity BEFORE TEST signal access.
    # --------------------------------------------------------

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            INDEX_PATH
        )

    index_sha = sha256_file(
        INDEX_PATH
    )

    print(
        f"Index SHA256 : {index_sha}"
    )

    if (
        index_sha
        != EXPECTED_INDEX_SHA256
    ):
        raise RuntimeError(
            "FINAL TEST INDEX HASH MISMATCH"
        )

    eval_df = pd.read_csv(
        INDEX_PATH
    )

    if len(eval_df) != EXPECTED_TEST_WINDOWS:
        raise RuntimeError(
            "TEST window count mismatch"
        )

    if set(
        eval_df["repeat"].unique()
    ) != {1}:
        raise RuntimeError(
            "TEST index repeat mismatch"
        )

    if set(
        eval_df["split"].unique()
    ) != {"test"}:
        raise RuntimeError(
            "TEST index split mismatch"
        )

    if (
        eval_df[
            "window_key_sha256"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate TEST window keys"
        )

    trial_cols = [
        "texture_id",
        "direction",
        "velocity",
        "force_mN",
        "repeat",
    ]

    trial_counts = (
        eval_df.groupby(
            trial_cols
        ).size()
    )

    if len(trial_counts) != EXPECTED_TEST_TRIALS:
        raise RuntimeError(
            "TEST trial count mismatch"
        )

    if not (
        trial_counts == 3
    ).all():
        raise RuntimeError(
            "Not exactly 3 windows/trial"
        )

    print(
        f"TEST trials  : {len(trial_counts)}"
    )

    print(
        f"TEST windows : {len(eval_df)}"
    )

    print(
        "INDEX CHECK  : PASS"
    )

    # --------------------------------------------------------
    # 2. Load final frozen checkpoint.
    # --------------------------------------------------------

    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model, params = build_model(
        ckpt,
        device
    )

    print(
        f"Checkpoint   : {CHECKPOINT}"
    )

    print(
        f"Best epoch   : {ckpt.get('epoch')}"
    )

    print(
        f"Parameters   : {params:,}"
    )

    print(
        "Model        : Final Visual-Physics Multi-FiLM"
    )

    # --------------------------------------------------------
    # 3. THIS IS THE POINT WHERE FINAL TEST IS OPENED.
    # --------------------------------------------------------

    print("")
    print(
        ">>> FINAL TEST SIGNAL ACCESS BEGINS <<<"
    )

    # Frozen final physical-condition configuration.
    # Frozen final Dataset reads this module-level setting
    # when appending condition channels.
    S.ACTIVE_CONDITION_MODE = "full"

    test_ds = S.FrozenValDataset(
        eval_df
    )

    if len(test_ds) != EXPECTED_TEST_WINDOWS:
        raise RuntimeError(
            "Dataset length mismatch"
        )

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # --------------------------------------------------------
    # 4. One forward pass over frozen TEST.
    #
    # Save all GT and predictions so plots can later be
    # reformatted without rerunning the model.
    # --------------------------------------------------------

    all_gt = []
    all_pred = []

    inference_seconds = 0.0

    model.eval()

    with torch.inference_mode():

        for batch_idx, (
            x,
            y,
            meta
        ) in enumerate(
            loader,
            start=1
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )

            t0 = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                pred = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )

            inference_seconds += (
                time.perf_counter()
                -
                t0
            )

            all_gt.append(
                y.numpy().astype(
                    np.float32,
                    copy=False
                )
            )

            all_pred.append(
                pred.detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )

            if (
                batch_idx % 10 == 0
                or
                batch_idx == len(loader)
            ):
                print(
                    f"TEST inference "
                    f"{batch_idx}/{len(loader)}"
                )

    gt_all = np.concatenate(
        all_gt,
        axis=0
    )

    pred_all = np.concatenate(
        all_pred,
        axis=0
    )

    if gt_all.shape != (
        EXPECTED_TEST_WINDOWS,
        FUTURE_N,
        3,
    ):
        raise RuntimeError(
            f"GT shape mismatch: {gt_all.shape}"
        )

    if pred_all.shape != gt_all.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: "
            f"{pred_all.shape}"
        )

    if not (
        np.isfinite(gt_all).all()
        and
        np.isfinite(pred_all).all()
    ):
        raise RuntimeError(
            "Non-finite GT/prediction"
        )

    np.savez_compressed(
        PRED_DIR
        / "final_test_predictions.npz",
        gt=gt_all,
        pred=pred_all,
        window_key_sha256=(
            eval_df[
                "window_key_sha256"
            ].astype(str).to_numpy()
        ),
    )

    # --------------------------------------------------------
    # 5. XYZ metrics.
    # Exactly same compute_metrics as frozen S evaluation.
    # --------------------------------------------------------

    metric_rows = []

    axes = ["X", "Y", "Z"]

    print("")
    print(
        "Computing XYZ metrics..."
    )

    for wi in range(
        EXPECTED_TEST_WINDOWS
    ):

        r = eval_df.iloc[wi]

        for ai, axis in enumerate(
            axes
        ):

            m = S.compute_metrics(
                gt_all[
                    wi,
                    :,
                    ai
                ],
                pred_all[
                    wi,
                    :,
                    ai
                ],
            )

            out = {
                "model":
                    "Final_VisualPhysics_MultiFiLM",

                "axis":
                    axis,

                "texture_id":
                    int(r["texture_id"]),

                "direction":
                    int(r["direction"]),

                "velocity":
                    int(r["velocity"]),

                "force_mN":
                    int(r["force_mN"]),

                "repeat":
                    int(r["repeat"]),

                "window_id":
                    int(r["window_id"]),

                "forecast_start":
                    int(r["forecast_start"]),

                "window_key_sha256":
                    str(
                        r["window_key_sha256"]
                    ),
            }

            out.update(m)

            metric_rows.append(
                out
            )

        if (
            (wi + 1) % 2000 == 0
            or
            wi + 1
            ==
            EXPECTED_TEST_WINDOWS
        ):
            print(
                f"XYZ metrics "
                f"{wi + 1}/"
                f"{EXPECTED_TEST_WINDOWS}"
            )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    if len(
        metrics_df
    ) != EXPECTED_METRIC_ROWS:
        raise RuntimeError(
            "Metric row count mismatch"
        )

    metrics_df.to_csv(
        OUT
        / "final_test_window_axis_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # 6. Robust median summaries.
    # --------------------------------------------------------

    summary_metrics = list(
        S.SUMMARY_METRICS
    )

    overall = {
        "model":
            "Final_VisualPhysics_MultiFiLM",

        "representation":
            "XYZ",

        "aggregation":
            "median_window_axis",
    }

    for col in summary_metrics:
        overall[col] = (
            robust_median(
                metrics_df[col]
            )
        )

    summary_df = pd.DataFrame(
        [overall]
    )

    summary_df.to_csv(
        OUT
        / "final_test_summary.csv",
        index=False
    )

    axis_rows = []

    for axis, g in metrics_df.groupby(
        "axis",
        sort=True
    ):

        row = {
            "model":
                "Final_VisualPhysics_MultiFiLM",

            "axis":
                axis,
        }

        for col in summary_metrics:
            row[col] = (
                robust_median(
                    g[col]
                )
            )

        axis_rows.append(
            row
        )

    axis_df = pd.DataFrame(
        axis_rows
    )

    axis_df.to_csv(
        OUT
        / "final_test_axis_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 7. GT-only visualization selection.
    # Prediction has NO role in selection.
    # --------------------------------------------------------

    (
        selected,
        strengths,
        median_target,
        p95_target,
    ) = select_visual_windows(
        eval_df,
        gt_all
    )

    vis_rows = []
    vis_npz = {}

    for name, wi in (
        selected.items()
    ):

        r = eval_df.iloc[
            wi
        ]

        gt = gt_all[
            wi
        ]

        pred = pred_all[
            wi
        ]

        gt321, pr321 = plot_dft321(
            name=name,
            gt=gt,
            pred=pred,
            out_path=(
                FIG_DIR
                / f"{name}_DFT321.png"
            ),
        )

        plot_xyz(
            name=name,
            gt=gt,
            pred=pred,
            meta=r,
            out_path=(
                FIG_DIR
                / f"{name}_XYZ.png"
            ),
        )

        vis_rows.append(
            {
                "selection_name":
                    name,

                "test_array_index":
                    int(wi),

                "texture_id":
                    int(r["texture_id"]),

                "direction":
                    int(r["direction"]),

                "velocity":
                    int(r["velocity"]),

                "force_mN":
                    int(r["force_mN"]),

                "repeat":
                    int(r["repeat"]),

                "window_id":
                    int(r["window_id"]),

                "forecast_start":
                    int(r["forecast_start"]),

                "window_key_sha256":
                    str(
                        r["window_key_sha256"]
                    ),

                "gt_dynamic_strength":
                    float(
                        strengths[wi]
                    ),
            }
        )

        vis_npz[
            f"{name}_gt_xyz"
        ] = gt

        vis_npz[
            f"{name}_pred_xyz"
        ] = pred

        vis_npz[
            f"{name}_gt_dft321"
        ] = gt321

        vis_npz[
            f"{name}_pred_dft321"
        ] = pr321

    pd.DataFrame(
        vis_rows
    ).to_csv(
        OUT
        / "visualization_windows.csv",
        index=False
    )

    np.savez_compressed(
        OUT
        / "visualization_signals.npz",
        **vis_npz
    )

    # --------------------------------------------------------
    # 8. Final integrity + report.
    # --------------------------------------------------------

    errors = []

    if len(metrics_df) != EXPECTED_METRIC_ROWS:
        errors.append(
            "metric row mismatch"
        )

    if (
        metrics_df[
            "window_key_sha256"
        ].nunique()
        !=
        EXPECTED_TEST_WINDOWS
    ):
        errors.append(
            "unique metric window mismatch"
        )

    if set(
        metrics_df["repeat"].unique()
    ) != {1}:
        errors.append(
            "metric repeat mismatch"
        )

    if len(selected) != 3:
        errors.append(
            "visualization selection mismatch"
        )

    if len(
        set(selected.values())
    ) != 3:
        errors.append(
            "visualization windows not unique"
        )

    required_figures = []

    for name in selected:
        required_figures.extend(
            [
                FIG_DIR
                / f"{name}_XYZ.png",

                FIG_DIR
                / f"{name}_DFT321.png",
            ]
        )

    for p in required_figures:
        if (
            not p.exists()
            or
            p.stat().st_size == 0
        ):
            errors.append(
                f"missing figure: {p}"
            )

    # Main reported values.
    o = summary_df.iloc[0]

    report = []

    report.append(
        "FINAL FROZEN REPEAT-1 TEST REPORT"
    )
    report.append(
        "=" * 78
    )
    report.append(
        f"Index SHA256      : {index_sha}"
    )
    report.append(
        f"Checkpoint        : {CHECKPOINT}"
    )
    report.append(
        f"Checkpoint epoch  : {ckpt.get('epoch')}"
    )
    report.append(
        f"Parameters        : {params}"
    )
    report.append(
        f"TEST trials       : {EXPECTED_TEST_TRIALS}"
    )
    report.append(
        f"TEST windows      : {EXPECTED_TEST_WINDOWS}"
    )
    report.append(
        f"Metric rows       : {len(metrics_df)}"
    )
    report.append(
        "Repeat            : 1"
    )
    report.append(
        "History           : 160 ms"
    )
    report.append(
        "Forecast          : 20 ms"
    )
    report.append(
        "Sampling          : 8000 Hz"
    )
    report.append(
        "TEST ACCESSED     : YES — FINAL ONLY"
    )

    report.append("")
    report.append(
        "XYZ ROBUST MEDIAN"
    )
    report.append(
        "-" * 78
    )

    report.append(
        f"RMSE              : {o['rmse']:.9f}"
    )
    report.append(
        f"MAE               : {o['mae']:.9f}"
    )
    report.append(
        f"r                 : {o['corr']:.9f}"
    )
    report.append(
        f"R2                : {o['r2']:.9f}"
    )
    report.append(
        f"dynamic-r         : {o['dynamic_corr']:.9f}"
    )
    report.append(
        f"dynamic-R2        : {o['dynamic_r2']:.9f}"
    )
    report.append(
        f"diff-r            : {o['diff_corr']:.9f}"
    )
    report.append(
        f"RMS-ratio         : {o['rms_ratio']:.9f}"
    )
    report.append(
        f"centroid error Hz : "
        f"{o['centroid_abs_error']:.9f}"
    )
    report.append(
        f"500-1000 logerr   : "
        f"{o['band_500_1000_log_error']:.9f}"
    )

    report.append("")
    report.append(
        "VISUALIZATION"
    )
    report.append(
        "-" * 78
    )
    report.append(
        f"GT median dynamic target : "
        f"{median_target:.9f}"
    )
    report.append(
        f"GT P95 dynamic target    : "
        f"{p95_target:.9f}"
    )

    for r in vis_rows:
        report.append(
            f"{r['selection_name']}: "
            f"texture={r['texture_id']}, "
            f"v={r['velocity']}, "
            f"dir={r['direction']}, "
            f"force={r['force_mN']}, "
            f"window={r['window_id']}, "
            f"D={r['gt_dynamic_strength']:.9f}"
        )

    report.append("")
    report.append(
        f"Inference total s : "
        f"{inference_seconds:.6f}"
    )
    report.append(
        f"Errors            : {len(errors)}"
    )

    if errors:
        for e in errors:
            report.append(
                f"ERROR: {e}"
            )

        decision = (
            "FAIL_FINAL_TEST"
        )
    else:
        decision = (
            "PASS_FINAL_TEST"
        )

    report.append("")
    report.append(
        f"FINAL DECISION: {decision}"
    )

    report_text = "\n".join(
        report
    )

    (
        OUT
        / "final_test_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8"
    )

    metadata = {
        "stage":
            "FINAL_FROZEN_REPEAT1_TEST",

        "test_accessed":
            True,

        "test_access":
            "FINAL_ONLY",

        "index_sha256":
            index_sha,

        "checkpoint":
            str(CHECKPOINT),

        "checkpoint_epoch":
            int(
                ckpt.get("epoch")
            ),

        "parameters":
            params,

        "test_trials":
            EXPECTED_TEST_TRIALS,

        "test_windows":
            EXPECTED_TEST_WINDOWS,

        "metric_rows":
            len(metrics_df),

        "history_n":
            HISTORY_N,

        "future_n":
            FUTURE_N,

        "fs":
            FS,

        "visualization_selection":
            "GT_only",

        "visualization_windows":
            vis_rows,

        "errors":
            errors,

        "decision":
            decision,
    }

    with open(
        OUT
        / "final_test_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("")
    print(report_text)
    print("")
    print(
        "Figures:"
    )

    for p in required_figures:
        print(
            f"  {p}"
        )


if __name__ == "__main__":
    main()
