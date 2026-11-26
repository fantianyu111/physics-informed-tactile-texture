from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


FS = 8000.0
HISTORY_N = 1280
FUTURE_N = 160

MANIFEST = Path(
    "cluster_H4C_results/H4C_cache_manifest.csv"
)

OUT_ROOT = Path(
    "cluster_FINAL_TEST_results"
)

OUT_INDEX = (
    OUT_ROOT
    / "final_test_eval_index.csv"
)

OUT_REPORT = (
    OUT_ROOT
    / "final_test_protocol_report.txt"
)

OUT_META = (
    OUT_ROOT
    / "final_test_protocol.json"
)


def make_key(row, window_id, start):
    s = (
        f"{int(row.texture_id)}|"
        f"{int(row.direction)}|"
        f"{int(row.velocity)}|"
        f"{int(row.force_mN)}|"
        f"{int(row.repeat)}|"
        f"{int(window_id)}|"
        f"{int(start)}"
    )
    return hashlib.sha256(
        s.encode("utf-8")
    ).hexdigest()


def main():

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        MANIFEST
    )

    # --------------------------------------------------------
    # Metadata-only selection.
    # DO NOT load cache_file/source_file.
    # --------------------------------------------------------

    test = df[
        (df["split"] == "test")
        &
        (df["repeat"] == 1)
    ].copy()

    errors = []

    if len(test) != 9440:
        errors.append(
            f"Expected 9440 TEST trials, got {len(test)}"
        )

    if set(test["repeat"].unique()) != {1}:
        errors.append(
            "TEST contains repeat other than 1"
        )

    if set(test["split"].unique()) != {"test"}:
        errors.append(
            "TEST contains non-test split"
        )

    if test[
        [
            "texture_id",
            "direction",
            "velocity",
            "force_mN",
            "repeat",
        ]
    ].duplicated().any():
        errors.append(
            "Duplicate TEST trial keys"
        )

    # Expected complete condition grid.
    if test["texture_id"].nunique() != 118:
        errors.append(
            "Expected 118 textures"
        )

    if set(
        test["velocity"].astype(int).unique()
    ) != {20, 30, 40, 50, 60}:
        errors.append(
            "Unexpected velocity levels"
        )

    if set(
        test["direction"].astype(int).unique()
    ) != {
        0, 45, 90, 135,
        180, 225, 270, 315
    }:
        errors.append(
            "Unexpected direction levels"
        )

    if set(
        test["force_mN"].astype(int).unique()
    ) != {500, 1000}:
        errors.append(
            "Unexpected force levels"
        )

    rows = []

    # --------------------------------------------------------
    # SAME deterministic three-position structure as frozen VAL:
    #
    # legal forecast starts:
    # HISTORY_N ... n_8k-FUTURE_N
    #
    # Choose approximately 10%, 50%, 90% through the legal
    # forecast-start interval.
    #
    # No signal is opened.
    # --------------------------------------------------------

    fractions = (
        0.10,
        0.50,
        0.90,
    )

    sort_cols = [
        "texture_id",
        "direction",
        "velocity",
        "force_mN",
        "repeat",
    ]

    test = test.sort_values(
        sort_cols
    ).reset_index(drop=True)

    for r in test.itertuples(
        index=False
    ):

        n = int(r.actual_n_8k)

        first_start = HISTORY_N
        last_start = n - FUTURE_N

        if last_start < first_start:
            errors.append(
                "Trial too short: "
                f"texture={r.texture_id}, "
                f"direction={r.direction}, "
                f"velocity={r.velocity}, "
                f"force={r.force_mN}"
            )
            continue

        span = (
            last_start
            -
            first_start
        )

        starts = [
            int(
                round(
                    first_start
                    +
                    q * span
                )
            )
            for q in fractions
        ]

        if len(set(starts)) != 3:
            errors.append(
                "Non-unique evaluation starts: "
                f"texture={r.texture_id}, "
                f"direction={r.direction}, "
                f"velocity={r.velocity}, "
                f"force={r.force_mN}"
            )
            continue

        for window_id, start in enumerate(
            starts
        ):

            rows.append(
                {
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
                        "test",

                    "window_id":
                        int(window_id),

                    "history_start":
                        int(
                            start
                            -
                            HISTORY_N
                        ),

                    "forecast_start":
                        int(start),

                    "future_end":
                        int(
                            start
                            +
                            FUTURE_N
                        ),

                    "history_n":
                        HISTORY_N,

                    "future_n":
                        FUTURE_N,

                    "history_ms":
                        160.0,

                    "horizon_ms":
                        20.0,

                    "n_8k":
                        n,

                    "source_file":
                        str(r.source_file),

                    "cache_file":
                        str(r.cache_file),

                    "window_key_sha256":
                        make_key(
                            r,
                            window_id,
                            start
                        ),
                }
            )

    out = pd.DataFrame(
        rows
    )

    expected_windows = (
        9440 * 3
    )

    if len(out) != expected_windows:
        errors.append(
            f"Expected {expected_windows} windows, "
            f"got {len(out)}"
        )

    if (
        len(out) > 0
        and
        out[
            "window_key_sha256"
        ].duplicated().any()
    ):
        errors.append(
            "Duplicate window_key_sha256"
        )

    if len(out) > 0:

        counts = (
            out.groupby(
                [
                    "texture_id",
                    "direction",
                    "velocity",
                    "force_mN",
                    "repeat",
                ]
            )
            .size()
        )

        if not (
            counts == 3
        ).all():
            errors.append(
                "Not every TEST trial has exactly 3 windows"
            )

        if not (
            out["future_end"]
            <=
            out["n_8k"]
        ).all():
            errors.append(
                "Future window exceeds trial boundary"
            )

        if not (
            out["history_start"]
            >= 0
        ).all():
            errors.append(
                "History window precedes trial boundary"
            )

    # --------------------------------------------------------
    # Important:
    # compare only the sampling STRUCTURE with VAL.
    # No VAL prediction/results are needed here.
    # --------------------------------------------------------

    val_path = Path(
        "cluster_B0_results/"
        "B0_val_eval_index.csv"
    )

    val = pd.read_csv(
        val_path
    )

    if len(val) != 5664:
        errors.append(
            f"Frozen VAL index changed: {len(val)} rows"
        )

    val_counts = (
        val.groupby(
            [
                "texture_id",
                "direction",
                "velocity",
                "force_mN",
                "repeat",
            ]
        )
        .size()
    )

    if not (
        val_counts == 3
    ).all():
        errors.append(
            "Frozen VAL no longer has 3 windows/trial"
        )

    out.to_csv(
        OUT_INDEX,
        index=False
    )

    index_sha256 = hashlib.sha256(
        OUT_INDEX.read_bytes()
    ).hexdigest()

    report = []

    report.append(
        "FINAL TEST PROTOCOL DRY-RUN"
    )
    report.append(
        "=" * 72
    )
    report.append(
        f"Manifest rows        : {len(df)}"
    )
    report.append(
        f"TEST trials          : {len(test)}"
    )
    report.append(
        f"TEST textures        : {test['texture_id'].nunique()}"
    )
    report.append(
        "TEST repeat          : 1"
    )
    report.append(
        "Windows per trial    : 3"
    )
    report.append(
        f"TEST windows         : {len(out)}"
    )
    report.append(
        f"Expected metric rows : {len(out) * 3}"
    )
    report.append(
        "History              : 160 ms / 1280 samples"
    )
    report.append(
        "Forecast             : 20 ms / 160 samples"
    )
    report.append(
        "Sampling             : 8000 Hz"
    )
    report.append(
        "Positions            : approx. 10%, 50%, 90%"
    )
    report.append(
        "Signal cache opened  : NO"
    )
    report.append(
        "Model loaded         : NO"
    )
    report.append(
        "Prediction performed : NO"
    )
    report.append(
        "TEST signal accessed : NO"
    )
    report.append(
        f"Index SHA256         : {index_sha256}"
    )
    report.append(
        f"Errors               : {len(errors)}"
    )

    if errors:
        report.append("")
        for e in errors:
            report.append(
                f"ERROR: {e}"
            )

        decision = (
            "FAIL_FINAL_TEST_PROTOCOL"
        )

    else:
        decision = (
            "PASS_FINAL_TEST_PROTOCOL"
        )

    report.append("")
    report.append(
        "VISUALIZATION RULES"
    )
    report.append(
        "-" * 72
    )
    report.append(
        "Selection uses GT signal only; prediction metrics are forbidden."
    )
    report.append(
        "Typical: GT dynamic strength nearest the TEST median."
    )
    report.append(
        "High-dynamic: GT dynamic strength nearest the TEST 95th percentile."
    )
    report.append(
        "Strongest-transient: maximum GT dynamic strength."
    )
    report.append(
        "Tie-break: ascending window_key_sha256."
    )
    report.append(
        "XYZ: GT and prediction, no smoothing or re-normalization."
    )
    report.append(
        "DFT321: same selected windows, validated DFT321 implementation."
    )

    report.append("")
    report.append(
        f"FINAL DECISION: {decision}"
    )

    text = "\n".join(
        report
    )

    OUT_REPORT.write_text(
        text,
        encoding="utf-8"
    )

    metadata = {
        "stage":
            "FINAL_TEST_PROTOCOL",

        "test_opened":
            False,

        "signal_cache_opened":
            False,

        "model_loaded":
            False,

        "prediction_performed":
            False,

        "repeat":
            1,

        "trial_count":
            int(len(test)),

        "windows_per_trial":
            3,

        "window_count":
            int(len(out)),

        "expected_metric_rows":
            int(len(out) * 3),

        "history_n":
            HISTORY_N,

        "future_n":
            FUTURE_N,

        "fs":
            FS,

        "sampling_positions":
            [
                0.10,
                0.50,
                0.90
            ],

        "index_sha256":
            index_sha256,

        "visualization": {
            "selection":
                "GT_only",

            "typical":
                "nearest median GT dynamic strength",

            "high_dynamic":
                "nearest 95th percentile GT dynamic strength",

            "strongest_transient":
                "maximum GT dynamic strength",

            "tie_break":
                "ascending window_key_sha256",

            "xyz_plot":
                True,

            "dft321_plot":
                True,
        },

        "errors":
            errors,

        "decision":
            decision,
    }

    with open(
        OUT_META,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(text)


if __name__ == "__main__":
    main()
