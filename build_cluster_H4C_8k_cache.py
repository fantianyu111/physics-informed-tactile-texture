#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[0]

MANIFEST = (
    BASE
    / "cluster_H1_results"
    / "H1_protocol_A_manifest.csv"
)

OUT = (
    BASE
    / "cluster_H4C_8k_cache"
)

REPORT_DIR = (
    BASE
    / "cluster_H4C_results"
)

FS = 8000.0
DT = 1.0 / FS

EXPECTED_TRIALS = 18880
EXPECTED_CHANNELS = 3

# If interrupted, existing valid cache files can be reused.
RESUME = True

# Print progress every N trials.
PRINT_EVERY = 200


def cache_relative_path(texture_id, source_file):

    """
    Cache hierarchy:
        cluster_H4C_8k_cache/
            000/
            001/
            ...
            117/

    Filename is derived from the original CSV stem.
    """

    texture_dir = f"{int(texture_id):03d}"

    stem = Path(
        str(source_file)
    ).stem

    return (
        Path(texture_dir)
        / f"{stem}.npy"
    )


def load_and_resample(source_file):

    """
    IMPORTANT:
    This reproduces the frozen H4B resampling rule:

        tu = np.arange(t[0], t[-1], 1/8000)

    Timestamp cleaning:
    - retain finite rows
    - retain first row
    - subsequent timestamp must be strictly increasing

    Interpolation:
    np.interp independently for X/Y/Z.

    Output:
        float32 [N, 3]
    """

    df = pd.read_csv(
        source_file,
        usecols=[
            "time",
            "X",
            "Y",
            "Z",
        ]
    )

    arr = df[
        [
            "time",
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    finite = np.all(
        np.isfinite(arr),
        axis=1
    )

    arr = arr[
        finite
    ]

    if len(arr) < 10:
        raise ValueError(
            "Too few finite samples."
        )

    t = arr[:, 0]
    xyz = arr[:, 1:4]

    keep = np.ones(
        len(t),
        dtype=bool
    )

    keep[1:] = (
        np.diff(t) > 0
    )

    t = t[
        keep
    ]

    xyz = xyz[
        keep
    ]

    if len(t) < 10:
        raise ValueError(
            "Too few samples after "
            "timestamp filtering."
        )

    tu = np.arange(
        t[0],
        t[-1],
        DT,
        dtype=np.float64
    )

    if len(tu) == 0:
        raise ValueError(
            "Uniform timestamp grid is empty."
        )

    yu = np.empty(
        (
            len(tu),
            EXPECTED_CHANNELS,
        ),
        dtype=np.float32
    )

    for axis in range(
        EXPECTED_CHANNELS
    ):

        yu[:, axis] = np.interp(
            tu,
            t,
            xyz[:, axis]
        ).astype(
            np.float32
        )

    return yu


def validate_array(
    arr,
    expected_n
):

    reasons = []

    if not isinstance(
        arr,
        np.ndarray
    ):
        reasons.append(
            "not_numpy_array"
        )

        return reasons

    if arr.ndim != 2:
        reasons.append(
            f"bad_ndim={arr.ndim}"
        )

        return reasons

    if arr.shape[1] != 3:
        reasons.append(
            f"bad_channels="
            f"{arr.shape[1]}"
        )

    if arr.shape[0] != expected_n:
        reasons.append(
            f"length_mismatch:"
            f"expected={expected_n},"
            f"actual={arr.shape[0]}"
        )

    if arr.dtype != np.float32:
        reasons.append(
            f"bad_dtype={arr.dtype}"
        )

    if not np.all(
        np.isfinite(arr)
    ):
        reasons.append(
            "nonfinite_values"
        )

    return reasons


def file_sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def main():

    print("=" * 78)
    print(
        "Cluster H4C — Frozen 8 kHz "
        "Acceleration Cache"
    )
    print("=" * 78)

    start_time = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Missing manifest: "
            f"{MANIFEST}"
        )

    manifest = pd.read_csv(
        MANIFEST
    )

    required = [
        "texture_id",
        "direction",
        "velocity",
        "force_mN",
        "repeat",
        "n_8k",
        "source_file",
        "split",
    ]

    missing = [
        c
        for c in required
        if c not in manifest.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing manifest columns: "
            f"{missing}"
        )

    if len(manifest) != EXPECTED_TRIALS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TRIALS} trials, "
            f"got {len(manifest)}"
        )

    if manifest[
        "source_file"
    ].duplicated().any():

        n_dup = int(
            manifest[
                "source_file"
            ].duplicated().sum()
        )

        raise RuntimeError(
            f"Duplicate source_file "
            f"entries: {n_dup}"
        )

    split_counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    expected_split = {
        "train": 7552,
        "val": 1888,
        "test": 9440,
    }

    if split_counts != expected_split:

        raise RuntimeError(
            f"Unexpected H1 split "
            f"counts: {split_counts}"
        )

    print(
        f"[H4C] Trials = "
        f"{len(manifest)}"
    )

    print(
        f"[H4C] Split = "
        f"{split_counts}"
    )

    print(
        f"[H4C] fs = "
        f"{FS:.0f} Hz"
    )

    print(
        f"[H4C] Resume = "
        f"{RESUME}"
    )

    print("")

    records = []
    errors = []

    created = 0
    reused = 0
    rebuilt = 0

    total_samples = 0
    total_bytes = 0

    for i, r in enumerate(
        manifest.itertuples(),
        start=1
    ):

        source_file = Path(
            str(
                r.source_file
            )
        )

        expected_n = int(
            r.n_8k
        )

        rel = cache_relative_path(
            r.texture_id,
            source_file
        )

        cache_file = (
            OUT
            / rel
        )

        cache_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        status = None
        rebuild_reason = ""

        try:

            if not source_file.exists():

                raise FileNotFoundError(
                    f"Missing source CSV: "
                    f"{source_file}"
                )

            use_existing = False

            if (
                RESUME
                and
                cache_file.exists()
            ):

                try:

                    existing = np.load(
                        cache_file,
                        mmap_mode="r",
                        allow_pickle=False
                    )

                    existing_errors = (
                        validate_array(
                            existing,
                            expected_n
                        )
                    )

                    if len(
                        existing_errors
                    ) == 0:

                        use_existing = True

                        status = "reused"

                        reused += 1

                    else:

                        rebuild_reason = (
                            ";".join(
                                existing_errors
                            )
                        )

                except Exception as e:

                    rebuild_reason = (
                        f"existing_cache_read_error:"
                        f"{type(e).__name__}:"
                        f"{e}"
                    )

            if use_existing:

                arr = np.load(
                    cache_file,
                    mmap_mode="r",
                    allow_pickle=False
                )

            else:

                existed_before = (
                    cache_file.exists()
                )

                arr_new = (
                    load_and_resample(
                        source_file
                    )
                )

                array_errors = (
                    validate_array(
                        arr_new,
                        expected_n
                    )
                )

                if array_errors:

                    raise RuntimeError(
                        ";".join(
                            array_errors
                        )
                    )

                # Atomic-ish write:
                # save temporary file, then replace.
                tmp_file = (
                    cache_file.parent
                    / (
                        cache_file.name
                        + ".tmp.npy"
                    )
                )

                np.save(
                    tmp_file,
                    arr_new,
                    allow_pickle=False
                )

                tmp_loaded = np.load(
                    tmp_file,
                    mmap_mode="r",
                    allow_pickle=False
                )

                tmp_errors = (
                    validate_array(
                        tmp_loaded,
                        expected_n
                    )
                )

                if tmp_errors:

                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass

                    raise RuntimeError(
                        "temporary_cache_validation:"
                        +
                        ";".join(
                            tmp_errors
                        )
                    )

                tmp_file.replace(
                    cache_file
                )

                arr = np.load(
                    cache_file,
                    mmap_mode="r",
                    allow_pickle=False
                )

                final_errors = (
                    validate_array(
                        arr,
                        expected_n
                    )
                )

                if final_errors:

                    raise RuntimeError(
                        "final_cache_validation:"
                        +
                        ";".join(
                            final_errors
                        )
                    )

                if existed_before:

                    status = "rebuilt"

                    rebuilt += 1

                else:

                    status = "created"

                    created += 1

            n_samples = int(
                arr.shape[0]
            )

            n_bytes = int(
                cache_file.stat().st_size
            )

            total_samples += (
                n_samples
            )

            total_bytes += (
                n_bytes
            )

            records.append({
                "texture_id":
                    int(r.texture_id),

                "direction":
                    int(r.direction),

                "velocity":
                    int(r.velocity),

                "force_mN":
                    int(r.force_mN),

                "repeat":
                    int(r.repeat),

                "split":
                    str(r.split),

                "source_file":
                    str(source_file),

                "cache_file":
                    str(cache_file),

                "expected_n_8k":
                    expected_n,

                "actual_n_8k":
                    n_samples,

                "channels":
                    int(
                        arr.shape[1]
                    ),

                "dtype":
                    str(
                        arr.dtype
                    ),

                "finite":
                    bool(
                        np.all(
                            np.isfinite(
                                arr
                            )
                        )
                    ),

                "bytes":
                    n_bytes,

                "status":
                    status,

                "rebuild_reason":
                    rebuild_reason,
            })

        except Exception as e:

            error_text = (
                f"trial={i}, "
                f"source={source_file}, "
                f"{type(e).__name__}: "
                f"{e}"
            )

            errors.append(
                error_text
            )

            print(
                "[H4C][ERROR] "
                + error_text,
                flush=True
            )

        if (
            i == 1
            or
            i % PRINT_EVERY == 0
            or
            i == len(manifest)
        ):

            elapsed = (
                time.time()
                -
                start_time
            )

            rate = (
                i / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                (
                    len(manifest) - i
                )
                /
                rate
                if rate > 0
                else np.nan
            )

            print(
                f"[H4C] "
                f"{i}/{len(manifest)} | "
                f"created={created} | "
                f"reused={reused} | "
                f"rebuilt={rebuilt} | "
                f"errors={len(errors)} | "
                f"{rate:.2f} trials/s | "
                f"ETA={remaining/60:.1f} min",
                flush=True
            )

    records_df = pd.DataFrame(
        records
    )

    records_df.to_csv(
        REPORT_DIR
        / "H4C_cache_manifest.csv",
        index=False
    )

    pd.DataFrame({
        "error": errors
    }).to_csv(
        REPORT_DIR
        / "H4C_errors.csv",
        index=False
    )

    # ========================================================
    # Full-cache integrity
    # ========================================================

    integrity_errors = []

    if len(records_df) != EXPECTED_TRIALS:

        integrity_errors.append(
            f"cache_manifest_rows="
            f"{len(records_df)}, "
            f"expected="
            f"{EXPECTED_TRIALS}"
        )

    if len(records_df) > 0:

        bad_length = records_df[
            records_df[
                "expected_n_8k"
            ]
            !=
            records_df[
                "actual_n_8k"
            ]
        ]

        if len(bad_length) > 0:

            integrity_errors.append(
                f"length_mismatch="
                f"{len(bad_length)}"
            )

        bad_channels = records_df[
            records_df[
                "channels"
            ]
            != 3
        ]

        if len(bad_channels) > 0:

            integrity_errors.append(
                f"bad_channels="
                f"{len(bad_channels)}"
            )

        bad_dtype = records_df[
            records_df[
                "dtype"
            ]
            != "float32"
        ]

        if len(bad_dtype) > 0:

            integrity_errors.append(
                f"bad_dtype="
                f"{len(bad_dtype)}"
            )

        bad_finite = records_df[
            records_df[
                "finite"
            ]
            != True
        ]

        if len(bad_finite) > 0:

            integrity_errors.append(
                f"nonfinite="
                f"{len(bad_finite)}"
            )

        duplicate_cache = int(
            records_df[
                "cache_file"
            ].duplicated().sum()
        )

        if duplicate_cache > 0:

            integrity_errors.append(
                f"duplicate_cache_file="
                f"{duplicate_cache}"
            )

    all_errors = (
        errors
        +
        integrity_errors
    )

    # ========================================================
    # Save cache metadata
    # ========================================================

    metadata = {
        "fs_hz":
            8000,

        "dtype":
            "float32",

        "channels":
            [
                "X",
                "Y",
                "Z",
            ],

        "timestamp_grid":
            "np.arange(t[0], t[-1], 1/8000)",

        "interpolation":
            "numpy.interp",

        "causal_demean":
            (
                "NOT stored in cache; "
                "must be applied per forecasting "
                "window from history only"
            ),

        "history_n":
            1280,

        "future_n":
            160,

        "stride_n":
            160,

        "cache_trials":
            int(
                len(records_df)
            ),

        "total_samples":
            int(
                total_samples
            ),

        "total_cache_bytes":
            int(
                total_bytes
            ),
    }

    with open(
        REPORT_DIR
        / "H4C_cache_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # Decision report
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )

    if len(records_df) > 0:

        min_n = int(
            records_df[
                "actual_n_8k"
            ].min()
        )

        median_n = int(
            records_df[
                "actual_n_8k"
            ].median()
        )

        max_n = int(
            records_df[
                "actual_n_8k"
            ].max()
        )

    else:

        min_n = -1
        median_n = -1
        max_n = -1

    cache_gib = (
        total_bytes
        /
        (1024 ** 3)
    )

    report = []

    report.append(
        "CLUSTER H4C FROZEN 8 KHZ "
        "ACCELERATION CACHE REPORT"
    )

    report.append(
        "=" * 78
    )

    report.append("")

    report.append(
        f"Source manifest: "
        f"{MANIFEST}"
    )

    report.append(
        f"Cache root: "
        f"{OUT}"
    )

    report.append("")

    report.append(
        "FROZEN PREPROCESSING"
    )

    report.append(
        "-" * 78
    )

    report.append(
        "Sampling rate: 8000 Hz"
    )

    report.append(
        "Timestamp grid: "
        "np.arange(t[0], t[-1], 1/8000)"
    )

    report.append(
        "Interpolation: np.interp "
        "independently on X/Y/Z"
    )

    report.append(
        "Stored dtype: float32"
    )

    report.append(
        "Stored shape: [N_8k, 3]"
    )

    report.append(
        "Causal demeaning is NOT "
        "baked into trial cache."
    )

    report.append(
        "It must remain a per-window "
        "history-only operation."
    )

    report.append("")

    report.append(
        "CACHE SUMMARY"
    )

    report.append(
        "-" * 78
    )

    report.append(
        f"Manifest trials: "
        f"{len(manifest)}"
    )

    report.append(
        f"Successful cache trials: "
        f"{len(records_df)}"
    )

    report.append(
        f"Created: {created}"
    )

    report.append(
        f"Reused: {reused}"
    )

    report.append(
        f"Rebuilt: {rebuilt}"
    )

    report.append(
        f"N_8k min/median/max: "
        f"{min_n}/"
        f"{median_n}/"
        f"{max_n}"
    )

    report.append(
        f"Total samples: "
        f"{total_samples}"
    )

    report.append(
        f"Cache size: "
        f"{cache_gib:.3f} GiB"
    )

    report.append(
        f"Elapsed: "
        f"{elapsed/60:.2f} min"
    )

    report.append("")

    report.append(
        "INTEGRITY"
    )

    report.append(
        "-" * 78
    )

    report.append(
        f"Raw processing errors: "
        f"{len(errors)}"
    )

    report.append(
        f"Integrity errors: "
        f"{len(integrity_errors)}"
    )

    report.append(
        f"Total errors: "
        f"{len(all_errors)}"
    )

    if len(records_df) > 0:

        report.append(
            "All successful caches have "
            "expected N_8k: "
            +
            (
                "PASS"
                if (
                    records_df[
                        "expected_n_8k"
                    ]
                    ==
                    records_df[
                        "actual_n_8k"
                    ]
                ).all()
                else
                "FAIL"
            )
        )

        report.append(
            "All successful caches have "
            "3 channels: "
            +
            (
                "PASS"
                if (
                    records_df[
                        "channels"
                    ]
                    ==
                    3
                ).all()
                else
                "FAIL"
            )
        )

        report.append(
            "All successful caches are "
            "float32: "
            +
            (
                "PASS"
                if (
                    records_df[
                        "dtype"
                    ]
                    ==
                    "float32"
                ).all()
                else
                "FAIL"
            )
        )

        report.append(
            "All successful caches finite: "
            +
            (
                "PASS"
                if (
                    records_df[
                        "finite"
                    ]
                    ==
                    True
                ).all()
                else
                "FAIL"
            )
        )

    report.append("")

    report.append(
        "IMPORTANT"
    )

    report.append(
        "-" * 78
    )

    report.append(
        "This cache stores resampled raw "
        "X/Y/Z acceleration only."
    )

    report.append(
        "It does NOT use future samples "
        "to compute normalization."
    )

    report.append(
        "For every forecasting window:"
    )

    report.append(
        "mu_h = mean(A_history, axis=time)"
    )

    report.append(
        "A_history_centered = "
        "A_history - mu_h"
    )

    report.append(
        "A_future_centered = "
        "A_future - mu_h"
    )

    report.append("")

    if len(all_errors) == 0:

        report.append(
            "FINAL H4C DECISION: "
            "PASS_FROZEN_8K_CACHE"
        )

    else:

        report.append(
            "FINAL H4C DECISION: FAIL"
        )

    report_text = "\n".join(
        report
    )

    (
        REPORT_DIR
        / "H4C_decision_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8"
    )

    # Save full error list including integrity errors.
    pd.DataFrame({
        "error": all_errors
    }).to_csv(
        REPORT_DIR
        / "H4C_all_errors.csv",
        index=False
    )

    print("")
    print(report_text)
    print("")

    print("=" * 78)
    print("[H4C] COMPLETE")
    print(
        f"[H4C] Cache = {OUT}"
    )
    print(
        f"[H4C] Report = {REPORT_DIR}"
    )
    print(
        f"[H4C] Errors = "
        f"{len(all_errors)}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
