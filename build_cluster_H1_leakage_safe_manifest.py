#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H1 — Cluster Leakage-Safe Trial Manifest
=========================================

Purpose
-------
Build the frozen trial-level split used by forecasting experiments.

IMPORTANT
---------
Split FIRST at trial level.
Window SECOND inside each split.

Never:
    all trials -> overlapping windows -> random split

Protocol A
----------
repeat = 1  -> TEST

repeat = 0:
    deterministic condition-balanced split
    -> TRAIN / VAL

The validation allocation is performed separately within every texture.
Each texture has 80 repeat-0 conditions:
    8 directions × 5 velocities × 2 forces

16 conditions/texture -> VAL
64 conditions/texture -> TRAIN

Therefore expected:
    TRAIN = 118 × 64 = 7552 trials
    VAL   = 118 × 16 = 1888 trials
    TEST  = 118 × 80 = 9440 trials

Total = 18880.

No forecasting windows are created here.
No history length/horizon is frozen here.
"""

from pathlib import Path
import re
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = (Path(__file__).resolve().parent / 'cluster_haptic_texture_dataset')

ACCEL_ROOT = ROOT / "sensor_data" / "accel"

OUT = (Path(__file__).resolve().parent / 'cluster_H1_results')

FS = 8000.0

EXPECTED_FILES = 18880
EXPECTED_TEXTURES = 118

EXPECTED_VELOCITIES = {
    20.0, 30.0, 40.0, 50.0, 60.0
}

EXPECTED_DIRECTIONS = {
    0.0, 45.0, 90.0, 135.0,
    180.0, 225.0, 270.0, 315.0
}

EXPECTED_FORCES = {
    500.0, 1000.0
}

EXPECTED_REPEATS = {
    0, 1
}

VAL_PER_TEXTURE = 16

BOUNDARY_VALUE = 4.0
BOUNDARY_ATOL = 1e-12


# ============================================================
# HELPERS
# ============================================================

def parse_filename(path):

    m = re.match(
        r"^(\d+)_(-?\d+)_([0-9.]+)_([0-9.]+)_(\d+)$",
        path.stem
    )

    if m is None:
        return None

    return {
        "texture_id": int(m.group(1)),
        "direction": float(m.group(2)),
        "velocity": float(m.group(3)),
        "force_mN": float(m.group(4)),
        "repeat": int(m.group(5)),
    }


def deterministic_score(
    texture_id,
    direction,
    velocity,
    force_mN
):
    """
    Stable deterministic pseudo-random score.

    Does NOT depend on Python hash randomization.
    """

    key = (
        f"{texture_id}_"
        f"{direction:.0f}_"
        f"{velocity:.0f}_"
        f"{force_mN:.0f}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16
    )


def inspect_trial(path):

    df = pd.read_csv(
        path,
        usecols=["time", "X", "Y", "Z"]
    )

    arr = df[
        ["time", "X", "Y", "Z"]
    ].to_numpy(
        dtype=np.float64
    )

    if len(arr) < 2:
        raise ValueError(
            "Trial has fewer than 2 samples."
        )

    finite = np.all(
        np.isfinite(arr),
        axis=1
    )

    if not np.all(finite):
        arr = arr[finite]

    if len(arr) < 2:
        raise ValueError(
            "Too few finite samples."
        )

    t = arr[:, 0]
    xyz = arr[:, 1:4]

    dt = np.diff(t)

    positive_dt = dt[
        dt > 0
    ]

    if len(positive_dt) == 0:
        raise ValueError(
            "No positive timestamp intervals."
        )

    duration = float(
        t[-1] - t[0]
    )

    if duration <= 0:
        raise ValueError(
            "Non-positive trial duration."
        )

    n_raw = int(
        len(t)
    )

    # Must match np.arange(t0, t_end, 1/fs)
    # used by the preprocessing scripts.
    n_8k = int(
        len(
            np.arange(
                t[0],
                t[-1],
                1.0 / FS
            )
        )
    )

    boundary_ratios = {}

    boundary_flags = {}

    for ai, axis in enumerate(
        ["X", "Y", "Z"]
    ):

        mask = np.isclose(
            np.abs(
                xyz[:, ai]
            ),
            BOUNDARY_VALUE,
            atol=BOUNDARY_ATOL,
            rtol=0.0
        )

        boundary_flags[axis] = bool(
            np.any(mask)
        )

        boundary_ratios[axis] = float(
            np.mean(mask)
        )

    return {
        "n_raw": n_raw,

        "duration_sec":
            duration,

        "mean_raw_fs":
            float(
                (n_raw - 1)
                /
                duration
            ),

        "median_dt":
            float(
                np.median(
                    positive_dt
                )
            ),

        "n_8k":
            n_8k,

        "boundary_flag_X":
            boundary_flags["X"],

        "boundary_flag_Y":
            boundary_flags["Y"],

        "boundary_flag_Z":
            boundary_flags["Z"],

        "boundary_ratio_X":
            boundary_ratios["X"],

        "boundary_ratio_Y":
            boundary_ratios["Y"],

        "boundary_ratio_Z":
            boundary_ratios["Z"],
    }


# ============================================================
# MANIFEST
# ============================================================

def build_manifest(files):

    rows = []
    errors = []

    total = len(files)

    for i, path in enumerate(
        files,
        start=1
    ):

        if (
            i == 1
            or i % 200 == 0
            or i == total
        ):
            print(
                f"[H1] Inspecting "
                f"{i}/{total}",
                flush=True
            )

        meta = parse_filename(
            path
        )

        if meta is None:

            errors.append({
                "source_file":
                    str(path),

                "error_type":
                    "FilenameParseError",

                "error":
                    "Could not parse filename."
            })

            continue

        try:

            trial_info = inspect_trial(
                path
            )

            row = {
                **meta,
                **trial_info,

                "source_file":
                    str(
                        path.resolve()
                    ),

                "condition_id":
                    (
                        f"T{meta['texture_id']:03d}_"
                        f"D{int(meta['direction']):03d}_"
                        f"V{int(meta['velocity']):02d}_"
                        f"F{int(meta['force_mN']):04d}"
                    )
            }

            rows.append(
                row
            )

        except Exception as e:

            errors.append({
                "source_file":
                    str(path),

                "error_type":
                    type(e).__name__,

                "error":
                    str(e)
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            errors,
            columns=[
                "source_file",
                "error_type",
                "error"
            ]
        )
    )


# ============================================================
# SPLIT
# ============================================================

def assign_protocol_a_split(df):

    df = df.copy()

    df["split"] = ""

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_mask = (
        df["repeat"] == 1
    )

    df.loc[
        test_mask,
        "split"
    ] = "test"

    # --------------------------------------------------------
    # TRAIN / VAL
    # repeat=0 only
    # --------------------------------------------------------

    repeat0 = df[
        df["repeat"] == 0
    ].copy()

    for texture_id, g in repeat0.groupby(
        "texture_id"
    ):

        if len(g) != 80:
            raise RuntimeError(
                f"Texture {texture_id}: "
                f"repeat0 has {len(g)} trials; "
                f"expected 80."
            )

        scores = []

        for idx, row in g.iterrows():

            score = deterministic_score(
                texture_id=int(
                    row["texture_id"]
                ),
                direction=float(
                    row["direction"]
                ),
                velocity=float(
                    row["velocity"]
                ),
                force_mN=float(
                    row["force_mN"]
                )
            )

            scores.append(
                (score, idx)
            )

        scores.sort(
            key=lambda x: x[0]
        )

        val_indices = [
            idx
            for _, idx
            in scores[
                :VAL_PER_TEXTURE
            ]
        ]

        train_indices = [
            idx
            for _, idx
            in scores[
                VAL_PER_TEXTURE:
            ]
        ]

        df.loc[
            val_indices,
            "split"
        ] = "val"

        df.loc[
            train_indices,
            "split"
        ] = "train"

    if np.any(
        df["split"] == ""
    ):
        raise RuntimeError(
            "Some trials were not assigned "
            "to any split."
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_manifest(df):

    problems = []

    # --------------------------------------------------------
    # Global structure
    # --------------------------------------------------------

    if len(df) != EXPECTED_FILES:
        problems.append(
            f"Manifest rows={len(df)}, "
            f"expected={EXPECTED_FILES}"
        )

    if (
        df["source_file"].nunique()
        != len(df)
    ):
        problems.append(
            "Duplicate source_file detected."
        )

    if (
        df["texture_id"].nunique()
        != EXPECTED_TEXTURES
    ):
        problems.append(
            "Texture count mismatch."
        )

    if set(
        df["velocity"].unique()
    ) != EXPECTED_VELOCITIES:
        problems.append(
            "Velocity set mismatch."
        )

    if set(
        df["direction"].unique()
    ) != EXPECTED_DIRECTIONS:
        problems.append(
            "Direction set mismatch."
        )

    if set(
        df["force_mN"].unique()
    ) != EXPECTED_FORCES:
        problems.append(
            "Force set mismatch."
        )

    if set(
        df["repeat"].unique()
    ) != EXPECTED_REPEATS:
        problems.append(
            "Repeat set mismatch."
        )

    # --------------------------------------------------------
    # Exact condition uniqueness
    # --------------------------------------------------------

    condition_repeat_cols = [
        "texture_id",
        "direction",
        "velocity",
        "force_mN",
        "repeat",
    ]

    duplicated = df.duplicated(
        condition_repeat_cols,
        keep=False
    )

    if duplicated.any():
        problems.append(
            "Duplicate texture/direction/"
            "velocity/force/repeat trials detected."
        )

    # --------------------------------------------------------
    # Expected split sizes
    # --------------------------------------------------------

    counts = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "train": 7552,
        "val": 1888,
        "test": 9440,
    }

    for split, expected in (
        expected_counts.items()
    ):

        actual = int(
            counts.get(
                split,
                0
            )
        )

        if actual != expected:
            problems.append(
                f"{split}: "
                f"{actual} != {expected}"
            )

    # --------------------------------------------------------
    # Repeat rule
    # --------------------------------------------------------

    if not (
        df.loc[
            df["split"] == "test",
            "repeat"
        ]
        == 1
    ).all():
        problems.append(
            "TEST contains repeat != 1."
        )

    if not (
        df.loc[
            df["split"].isin(
                ["train", "val"]
            ),
            "repeat"
        ]
        == 0
    ).all():
        problems.append(
            "TRAIN/VAL contains repeat != 0."
        )

    # --------------------------------------------------------
    # Every texture represented in every split
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        n_texture = (
            df.loc[
                df["split"] == split,
                "texture_id"
            ]
            .nunique()
        )

        if (
            n_texture
            != EXPECTED_TEXTURES
        ):
            problems.append(
                f"{split}: "
                f"only {n_texture} textures."
            )

    # --------------------------------------------------------
    # Every texture expected counts
    # --------------------------------------------------------

    expected_per_texture = {
        "train": 64,
        "val": 16,
        "test": 80,
    }

    for split, expected in (
        expected_per_texture.items()
    ):

        counts_tex = (
            df[
                df["split"] == split
            ]
            .groupby(
                "texture_id"
            )
            .size()
        )

        if not (
            counts_tex == expected
        ).all():

            bad = counts_tex[
                counts_tex != expected
            ]

            problems.append(
                f"{split}: per-texture "
                f"count mismatch: "
                f"{bad.to_dict()}"
            )

    # --------------------------------------------------------
    # No exact source-file leakage
    # --------------------------------------------------------

    split_sets = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_sets[split] = set(
            df.loc[
                df["split"] == split,
                "source_file"
            ]
        )

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for a, b in pairs:

        overlap = (
            split_sets[a]
            &
            split_sets[b]
        )

        if overlap:
            problems.append(
                f"Source-file leakage "
                f"{a}<->{b}: "
                f"{len(overlap)} files."
            )

    return problems


# ============================================================
# SUMMARIES
# ============================================================

def build_split_summary(df):

    rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        g = df[
            df["split"] == split
        ]

        rows.append({
            "split":
                split,

            "n_trials":
                len(g),

            "n_textures":
                g[
                    "texture_id"
                ].nunique(),

            "n_directions":
                g[
                    "direction"
                ].nunique(),

            "n_velocities":
                g[
                    "velocity"
                ].nunique(),

            "n_forces":
                g[
                    "force_mN"
                ].nunique(),

            "n_repeats":
                g[
                    "repeat"
                ].nunique(),

            "duration_sec_sum":
                g[
                    "duration_sec"
                ].sum(),

            "duration_sec_median":
                g[
                    "duration_sec"
                ].median(),

            "n8k_sum":
                int(
                    g[
                        "n_8k"
                    ].sum()
                ),

            "boundary_trials_X":
                int(
                    g[
                        "boundary_flag_X"
                    ].sum()
                ),

            "boundary_trials_Y":
                int(
                    g[
                        "boundary_flag_Y"
                    ].sum()
                ),

            "boundary_trials_Z":
                int(
                    g[
                        "boundary_flag_Z"
                    ].sum()
                ),
        })

    return pd.DataFrame(
        rows
    )


def build_condition_summary(df):

    rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        g = df[
            df["split"] == split
        ]

        for dimension in [
            "velocity",
            "direction",
            "force_mN"
        ]:

            counts = (
                g.groupby(
                    dimension
                )
                .size()
            )

            for value, count in (
                counts.items()
            ):

                rows.append({
                    "split":
                        split,

                    "dimension":
                        dimension,

                    "value":
                        value,

                    "count":
                        int(count)
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# REPORT
# ============================================================

def build_report(
    manifest,
    split_summary,
    condition_summary,
    errors,
    problems
):

    lines = []

    lines.append(
        "CLUSTER H1 LEAKAGE-SAFE "
        "TRIAL MANIFEST REPORT"
    )

    lines.append("=" * 78)
    lines.append("")

    lines.append(
        f"Acceleration root: "
        f"{ACCEL_ROOT}"
    )

    lines.append(
        f"Acceleration trials: "
        f"{len(manifest)}"
    )

    lines.append(
        f"Errors: {len(errors)}"
    )

    lines.append(
        f"Frozen preprocessing fs: "
        f"{FS:.0f} Hz"
    )

    lines.append("")

    lines.append(
        "Protocol A:"
    )

    lines.append(
        "  repeat=1 -> TEST"
    )

    lines.append(
        "  repeat=0 -> TRAIN / VAL"
    )

    lines.append(
        "  validation allocation is "
        "deterministic and performed "
        "within each texture"
    )

    lines.append("")

    lines.append(
        "CRITICAL RULE:"
    )

    lines.append(
        "  Trial split occurs BEFORE "
        "forecasting-window generation."
    )

    lines.append(
        "  Overlapping windows must never "
        "be randomly split across datasets."
    )

    lines.append("")

    lines.append("-" * 78)
    lines.append(
        "1. SPLIT SUMMARY"
    )
    lines.append("-" * 78)

    for _, r in (
        split_summary.iterrows()
    ):

        lines.append(
            f"{r['split'].upper()}: "
            f"trials={int(r['n_trials'])}, "
            f"textures={int(r['n_textures'])}, "
            f"directions={int(r['n_directions'])}, "
            f"velocities={int(r['n_velocities'])}, "
            f"forces={int(r['n_forces'])}, "
            f"repeats={int(r['n_repeats'])}, "
            f"duration={r['duration_sec_sum'] / 3600:.3f} h"
        )

    lines.append("")

    lines.append("-" * 78)
    lines.append(
        "2. BOUNDARY TRIAL DISTRIBUTION"
    )
    lines.append("-" * 78)

    for _, r in (
        split_summary.iterrows()
    ):

        lines.append(
            f"{r['split'].upper()}: "
            f"X={int(r['boundary_trials_X'])}, "
            f"Y={int(r['boundary_trials_Y'])}, "
            f"Z={int(r['boundary_trials_Z'])}"
        )

    lines.append("")

    lines.append("-" * 78)
    lines.append(
        "3. LEAKAGE / STRUCTURE CHECK"
    )
    lines.append("-" * 78)

    if problems:

        lines.append(
            "STATUS: FAIL"
        )

        for p in problems:
            lines.append(
                f"  - {p}"
            )

    else:

        lines.append(
            "Manifest rows: PASS"
        )

        lines.append(
            "Unique source files: PASS"
        )

        lines.append(
            "Condition-repeat uniqueness: PASS"
        )

        lines.append(
            "Expected split sizes: PASS"
        )

        lines.append(
            "Repeat isolation: PASS"
        )

        lines.append(
            "118-texture coverage: PASS"
        )

        lines.append(
            "Per-texture split counts: PASS"
        )

        lines.append(
            "Source-file overlap: 0 — PASS"
        )

    lines.append("")

    lines.append("-" * 78)
    lines.append(
        "4. H1 DECISION"
    )
    lines.append("-" * 78)

    if (
        len(errors) == 0
        and len(problems) == 0
    ):

        decision = (
            "PASS_LEAKAGE_SAFE_PROTOCOL_A"
        )

    else:

        decision = (
            "FAIL / MANUAL REVIEW REQUIRED"
        )

    lines.append("")
    lines.append(
        f"FINAL H1 DECISION: "
        f"{decision}"
    )
    lines.append("")

    if decision == (
        "PASS_LEAKAGE_SAFE_PROTOCOL_A"
    ):

        lines.append(
            "Protocol A trial-level split "
            "is ready for forecasting "
            "window generation."
        )

        lines.append("")

        lines.append(
            "This does NOT freeze:"
        )

        lines.append(
            "  - history length"
        )

        lines.append(
            "  - forecast horizon"
        )

        lines.append(
            "  - window stride"
        )

        lines.append(
            "  - deep model architecture"
        )

        lines.append("")

        lines.append(
            "Next:"
        )

        lines.append(
            "H2/H3 forecasting predictability "
            "with leakage-safe windows."
        )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "Cluster H1 — Leakage-Safe "
        "Trial Manifest"
    )
    print("=" * 78)

    print(
        f"[H1] ACCEL_ROOT = "
        f"{ACCEL_ROOT}",
        flush=True
    )

    if not ACCEL_ROOT.exists():

        raise FileNotFoundError(
            ACCEL_ROOT
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        ACCEL_ROOT.glob(
            "*/*.csv"
        )
    )

    print(
        f"[H1] Files discovered: "
        f"{len(files)}",
        flush=True
    )

    if len(files) != EXPECTED_FILES:

        raise RuntimeError(
            f"Expected {EXPECTED_FILES}, "
            f"found {len(files)}."
        )

    manifest, errors = (
        build_manifest(
            files
        )
    )

    print(
        f"[H1] Valid trials: "
        f"{len(manifest)}",
        flush=True
    )

    print(
        f"[H1] Errors: "
        f"{len(errors)}",
        flush=True
    )

    if len(errors) > 0:

        errors.to_csv(
            OUT /
            "H1_errors.csv",
            index=False
        )

        raise RuntimeError(
            "Trial inspection errors found. "
            "See H1_errors.csv."
        )

    print(
        "[H1] Assigning Protocol A...",
        flush=True
    )

    manifest = (
        assign_protocol_a_split(
            manifest
        )
    )

    problems = (
        validate_manifest(
            manifest
        )
    )

    split_summary = (
        build_split_summary(
            manifest
        )
    )

    condition_summary = (
        build_condition_summary(
            manifest
        )
    )

    # Stable ordering
    manifest = manifest.sort_values(
        [
            "split",
            "texture_id",
            "direction",
            "velocity",
            "force_mN",
            "repeat"
        ]
    ).reset_index(
        drop=True
    )

    # Save master manifest
    manifest.to_csv(
        OUT /
        "H1_protocol_A_manifest.csv",
        index=False
    )

    # Save split-specific manifests
    for split in [
        "train",
        "val",
        "test"
    ]:

        manifest[
            manifest["split"]
            ==
            split
        ].to_csv(
            OUT /
            f"H1_{split}_manifest.csv",
            index=False
        )

    split_summary.to_csv(
        OUT /
        "H1_split_summary.csv",
        index=False
    )

    condition_summary.to_csv(
        OUT /
        "H1_condition_summary.csv",
        index=False
    )

    errors.to_csv(
        OUT /
        "H1_errors.csv",
        index=False
    )

    report = build_report(
        manifest=manifest,
        split_summary=split_summary,
        condition_summary=condition_summary,
        errors=errors,
        problems=problems
    )

    report_path = (
        OUT /
        "H1_decision_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    print("")
    print(report)
    print("")

    print("=" * 78)
    print(
        "[H1] COMPLETE"
    )

    print(
        f"[H1] Results: {OUT}"
    )

    print(
        f"[H1] Problems: "
        f"{len(problems)}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
