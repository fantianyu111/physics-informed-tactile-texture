#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster Final — Visual Gated Single-FiLM

Frozen task:
    fs       = 8000 Hz
    history  = 160 ms / 1280 samples
    horizon  = 20 ms / 160 samples
    stride   = 20 ms / 160 samples

Training:
    H4C TRAIN cache only.

Validation:
    B0 frozen VAL evaluation index only.

TEST:
    NEVER ACCESSED.

This is deliberately a plain deep-learning baseline:
    raw centered tri-axis history
    -> Plain temporal-patch Transformer
    -> final hidden state
    -> Linear(128, 160*3)
    -> future tri-axis waveform

No:
    physical prior
    spectral branch
    attention
    persistence residual
    special axis weighting
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from scipy.signal import welch

warnings.filterwarnings("ignore")


# ============================================================
# FROZEN CONFIG
# ============================================================

BASE = Path(__file__).resolve().parents[0]

H4C_MANIFEST = (
    BASE
    / "cluster_H4C_results"
    / "H4C_cache_manifest.csv"
)

B0_VAL_INDEX = (
    BASE
    / "cluster_B0_results"
    / "B0_val_eval_index.csv"
)

OUT_ROOT = (
    BASE
    / "cluster_Final_singlefilm_results"
)

FS = 8000.0

HISTORY_N = 1280
FUTURE_N = 160
STRIDE_N = 160

HISTORY_MS = 160.0
HORIZON_MS = 20.0
STRIDE_MS = 20.0

AXES = ["X", "Y", "Z"]

BANDS = [
    (100.0, 500.0, "100_500"),
    (500.0, 1000.0, "500_1000"),
    (1000.0, 2000.0, "1000_2000"),
]

EPS = 1e-12

VISUAL_FEATURE_PATH = (
    Path(__file__).resolve().parent
    / "cluster_visual_features"
    / "swin_t_multistage_cropmean_features.npz"
)



# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# TRAIN WINDOW INDEX
# ============================================================

def build_train_windows(
    manifest,
    max_windows=None
):

    train = (
        manifest[
            manifest["split"]
            .astype(str)
            .str.lower()
            == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows = []

    for trial_id, r in enumerate(
        train.itertuples()
    ):

        n = int(
            r.actual_n_8k
        )

        first_start = HISTORY_N
        last_start = n - FUTURE_N

        if last_start < first_start:
            continue

        starts = np.arange(
            first_start,
            last_start + 1,
            STRIDE_N,
            dtype=np.int64
        )

        for start in starts:

            rows.append(
                (
                    trial_id,
                    int(start)
                )
            )

    if len(rows) == 0:
        raise RuntimeError(
            "No TRAIN windows."
        )

    windows = np.asarray(
        rows,
        dtype=np.int64
    )

    if (
        max_windows is not None
        and
        len(windows) > max_windows
    ):

        # Deterministic evenly distributed smoke subset.
        idx = np.linspace(
            0,
            len(windows) - 1,
            max_windows
        )

        idx = np.round(
            idx
        ).astype(np.int64)

        windows = windows[idx]

    return train, windows


# Active condition mode for controlled ablation.
# Set once in main() before DataLoader construction.
ACTIVE_CONDITION_MODE = None


# ============================================================
# PHYSICAL CONDITION ENCODING
# ============================================================

def encode_physical_condition(
    velocity,
    direction,
    force_mN,
    condition_mode
):
    """
    Physical-condition ablation.

    velocity:
        [v_norm]

    velocity_direction:
        [v_norm, sin(theta), cos(theta)]

    full:
        [v_norm, sin(theta), cos(theta), F_norm]
    """

    v = (
        float(velocity) - 40.0
    ) / 20.0

    theta = np.deg2rad(
        float(direction)
    )

    f = (
        float(force_mN) - 750.0
    ) / 250.0

    if condition_mode == "velocity":

        values = [
            v
        ]

    elif condition_mode == "velocity_direction":

        values = [
            v,
            np.sin(theta),
            np.cos(theta),
        ]

    elif condition_mode == "full":

        values = [
            v,
            np.sin(theta),
            np.cos(theta),
            f,
        ]

    else:
        raise ValueError(
            f"Unknown condition_mode="
            f"{condition_mode}"
        )

    return np.asarray(
        values,
        dtype=np.float32
    )

def append_condition_channels(
    history,
    velocity,
    direction,
    force_mN,
    condition_mode
):
    """
    Append selected physical conditions
    as constant channels.
    """

    cond = encode_physical_condition(
        velocity,
        direction,
        force_mN,
        condition_mode
    )

    cond_seq = np.broadcast_to(
        cond[None, :],
        (
            history.shape[0],
            cond.shape[0]
        )
    ).copy()

    return np.concatenate(
        [
            history,
            cond_seq
        ],
        axis=1
    )


class TrainWindowDataset(Dataset):

    def __init__(
        self,
        train_trials,
        windows
    ):

        self.train_trials = (
            train_trials
        )

        self.windows = windows

        self._cache_trial = None
        self._cache_signal = None

    def __len__(self):

        return len(self.windows)

    def _load_trial(
        self,
        trial_id
    ):

        if (
            self._cache_trial
            == trial_id
            and
            self._cache_signal
            is not None
        ):
            return self._cache_signal

        row = (
            self.train_trials.iloc[
                int(trial_id)
            ]
        )

        signal = np.load(
            str(row["cache_file"]),
            mmap_mode="r"
        )

        if (
            signal.ndim != 2
            or
            signal.shape[1] != 3
        ):
            raise RuntimeError(
                f"Bad cache shape "
                f"{signal.shape}"
            )

        self._cache_trial = (
            trial_id
        )

        self._cache_signal = (
            signal
        )

        return signal

    def __getitem__(
        self,
        idx
    ):

        trial_id, start = (
            self.windows[idx]
        )

        signal = self._load_trial(
            int(trial_id)
        )

        start = int(start)

        history = np.asarray(
            signal[
                start - HISTORY_N:
                start
            ],
            dtype=np.float32
        ).copy()

        future = np.asarray(
            signal[
                start:
                start + FUTURE_N
            ],
            dtype=np.float32
        ).copy()

        if history.shape != (
            HISTORY_N,
            3
        ):
            raise RuntimeError(
                f"History shape "
                f"{history.shape}"
            )

        if future.shape != (
            FUTURE_N,
            3
        ):
            raise RuntimeError(
                f"Future shape "
                f"{future.shape}"
            )

        # H0 frozen causal preprocessing.
        mu_h = history.mean(
            axis=0,
            keepdims=True
        )

        history -= mu_h
        future -= mu_h

        row = self.train_trials.iloc[
            int(trial_id)
        ]

        history_cond = append_condition_channels(
            history=history,
            velocity=row["velocity"],
            direction=row["direction"],
            force_mN=row["force_mN"],
            condition_mode=ACTIVE_CONDITION_MODE,
        )

        texture_id_channel = np.full(
            (history.shape[0], 1),
            float(row["texture_id"]),
            dtype=np.float32
        )

        history_cond = np.concatenate(
            [
                history_cond,
                texture_id_channel
            ],
            axis=1
        )

        return (
            torch.from_numpy(history_cond),
            torch.from_numpy(future)
        )


class FrozenValDataset(Dataset):

    def __init__(
        self,
        eval_df,
        max_windows=None
    ):

        if max_windows is not None:
            eval_df = (
                eval_df
                .iloc[:max_windows]
                .copy()
            )

        self.df = (
            eval_df
            .reset_index(drop=True)
        )

        self._cache_file = None
        self._cache_signal = None

    def __len__(self):

        return len(self.df)

    def _load(
        self,
        cache_file
    ):

        cache_file = str(
            cache_file
        )

        if (
            cache_file
            == self._cache_file
            and
            self._cache_signal
            is not None
        ):
            return self._cache_signal

        signal = np.load(
            cache_file,
            mmap_mode="r"
        )

        self._cache_file = cache_file
        self._cache_signal = signal

        return signal

    def __getitem__(
        self,
        idx
    ):

        r = self.df.iloc[idx]

        signal = self._load(
            r["cache_file"]
        )

        start = int(
            r["forecast_start"]
        )

        history = np.asarray(
            signal[
                start - HISTORY_N:
                start
            ],
            dtype=np.float32
        ).copy()

        future = np.asarray(
            signal[
                start:
                start + FUTURE_N
            ],
            dtype=np.float32
        ).copy()

        mu_h = history.mean(
            axis=0,
            keepdims=True
        )

        history -= mu_h
        future -= mu_h

        history_cond = append_condition_channels(
            history=history,
            velocity=r["velocity"],
            direction=r["direction"],
            force_mN=r["force_mN"],
            condition_mode=ACTIVE_CONDITION_MODE,
        )

        texture_id_channel = np.full(
            (history.shape[0], 1),
            float(r["texture_id"]),
            dtype=np.float32
        )

        history_cond = np.concatenate(
            [
                history_cond,
                texture_id_channel
            ],
            axis=1
        )

        meta = {
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
                    r[
                        "window_key_sha256"
                    ]
                ),
        }

        return (
            torch.from_numpy(history_cond),
            torch.from_numpy(future),
            meta
        )


# ============================================================
# TRIAL-LOCAL TRAIN SAMPLER
# ============================================================

class TrialGroupedSampler(Sampler):

    """
    Randomize trial order every epoch while
    keeping windows from one trial together.

    This preserves every training window exactly
    once/epoch while greatly improving .npy cache
    locality.
    """

    def __init__(
        self,
        windows,
        seed
    ):

        self.windows = windows
        self.seed = seed
        self.epoch = 0

        groups = {}

        for dataset_idx, (
            trial_id,
            _
        ) in enumerate(windows):

            groups.setdefault(
                int(trial_id),
                []
            ).append(
                dataset_idx
            )

        self.groups = groups
        self.trial_ids = list(
            groups.keys()
        )

    def set_epoch(
        self,
        epoch
    ):

        self.epoch = epoch

    def __iter__(self):

        rng = np.random.default_rng(
            self.seed
            +
            self.epoch
        )

        trials = np.asarray(
            self.trial_ids,
            dtype=np.int64
        )

        rng.shuffle(trials)

        indices = []

        for trial_id in trials:

            g = np.asarray(
                self.groups[
                    int(trial_id)
                ],
                dtype=np.int64
            )

            # Randomize windows inside trial.
            rng.shuffle(g)

            indices.extend(
                g.tolist()
            )

        return iter(indices)

    def __len__(self):

        return len(self.windows)


# ============================================================
# MODEL
# ============================================================

class ConditionedTransformer(nn.Module):
    """
    Fast screening model for physical-condition fusion.

    Input:
        x[:, :, 0:3]
            causal-centered acceleration

        x[:, :, 3:7]
            [v_norm, sin(theta), cos(theta), F_norm]

    Fusion:
        additive
        or
        FiLM

    No spectral branch/loss.
    No adaptive gate.
    """

    def __init__(
        self,
        hidden_dim=96,
        num_layers=3,
        dropout=0.1,
        patch_len=20,
        nhead=4,
        fusion="additive",
        condition_dim=4,
    ):
        super().__init__()

        if fusion not in {
            "additive",
            "film",
            "multifilm"
        }:
            raise ValueError(
                f"Unknown fusion={fusion}"
            )

        if HISTORY_N % patch_len != 0:
            raise ValueError(
                f"HISTORY_N={HISTORY_N} must be "
                f"divisible by patch_len={patch_len}"
            )

        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be "
                f"divisible by nhead={nhead}"
            )

        self.fusion = fusion
        self.patch_len = patch_len

        self.num_patches = (
            HISTORY_N // patch_len
        )

        # ------------------------------
        # Temporal patch embedding
        # ------------------------------

        self.patch_proj = nn.Linear(
            patch_len * 3,
            hidden_dim
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                hidden_dim
            )
        )

        # ------------------------------
        # Physical condition encoder
        #
        # [v, sin(theta), cos(theta), F]
        #             4 -> 32 -> hidden
        # ------------------------------

        self.condition_dim = condition_dim

        self.physics_encoder = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.GELU(),
            nn.Linear(
                32,
                hidden_dim
            ),
        )

        # ------------------------------
        # I7 multistage Swin visual branch
        # ------------------------------

        if not VISUAL_FEATURE_PATH.exists():
            raise RuntimeError(
                f"Missing visual feature file: "
                f"{VISUAL_FEATURE_PATH}"
            )

        visual_npz = np.load(
            VISUAL_FEATURE_PATH
        )

        visual_table = np.asarray(
            visual_npz["features"],
            dtype=np.float32
        )

        if visual_table.shape != (118, 1344):
            raise RuntimeError(
                f"Expected visual table "
                f"(118, 1344), got "
                f"{visual_table.shape}"
            )

        if not np.isfinite(
            visual_table
        ).all():
            raise RuntimeError(
                "Non-finite visual features."
            )

        self.register_buffer(
            "visual_table",
            torch.from_numpy(
                visual_table
            ),
            persistent=True
        )

        self.visual_proj = nn.Sequential(
            nn.Linear(
                1344,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # Learnable weak visual residual gate.
        #
        # sigmoid(-4) ~= 0.018, so the model starts
        # very close to the physics-only S1 condition.
        self.visual_gate_logit = nn.Parameter(
            torch.tensor(-4.0)
        )

        # FiLM-specific head.
        #
        # Keep the common physics encoder
        # identical between Additive and FiLM.
        #
        # gamma/beta start near zero so that
        # early training remains close to the
        # plain temporal representation.

        if self.fusion == "film":

            self.film_head = nn.Linear(
                hidden_dim,
                hidden_dim * 2
            )

            nn.init.zeros_(
                self.film_head.weight
            )

            nn.init.zeros_(
                self.film_head.bias
            )

            self.film_heads = None

        elif self.fusion == "multifilm":

            # One independent FiLM head per
            # Transformer layer. All heads use
            # the same physical representation p.
            self.film_head = None

            self.film_heads = nn.ModuleList([
                nn.Linear(
                    hidden_dim,
                    hidden_dim * 2
                )
                for _ in range(num_layers)
            ])

            # Start close to the plain Transformer.
            for head in self.film_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        else:
            self.film_head = None
            self.film_heads = None

        # ------------------------------
        # Same B3 Transformer backbone
        # ------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=(
                    hidden_dim * 2
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.head = nn.Linear(
            hidden_dim,
            FUTURE_N * 3
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02
        )

    def forward(
        self,
        x
    ):
        # Expected:
        # [B, 1280, 7]

        B, T, C = x.shape

        if (
            T != HISTORY_N
            or
            C != (3 + self.condition_dim + 1)
        ):
            raise ValueError(
                f"Expected "
                f"[B,{HISTORY_N},{3 + self.condition_dim + 1}], "
                f"got {tuple(x.shape)}"
            )

        # --------------------------------
        # Split acceleration / condition
        # --------------------------------

        accel = x[:, :, :3]

        # Condition channels are constant
        # across the history window.
        cond = x[
            :,
            0,
            3:(3 + self.condition_dim)
        ]

        texture_id = (
            x[
                :,
                0,
                3 + self.condition_dim
            ]
            .round()
            .long()
        )

        if (
            torch.any(texture_id < 0)
            or
            torch.any(texture_id >= 118)
        ):
            raise RuntimeError(
                "texture_id out of range"
            )

        # --------------------------------
        # Temporal tokens
        # --------------------------------

        accel = accel.reshape(
            B,
            self.num_patches,
            self.patch_len,
            3,
        )

        accel = accel.reshape(
            B,
            self.num_patches,
            self.patch_len * 3,
        )

        z = self.patch_proj(
            accel
        )

        z = (
            z
            +
            self.pos_embed
        )

        # --------------------------------
        # Physical representation
        # --------------------------------

        p_phys = self.physics_encoder(
            cond
        )

        visual_raw = self.visual_table[
            texture_id
        ]

        p_visual = self.visual_proj(
            visual_raw
        )

        visual_gate = torch.sigmoid(
            self.visual_gate_logit
        )

        p = (
            p_phys
            +
            visual_gate * p_visual
        )

        # --------------------------------
        # C1: additive condition injection
        # --------------------------------

        if self.fusion == "additive":

            z = (
                z
                +
                p[:, None, :]
            )

        # --------------------------------
        # C2: Physics-FiLM
        #
        # z' = (1 + gamma) * z + beta
        # --------------------------------

        elif self.fusion == "film":

            gamma_beta = (
                self.film_head(p)
            )

            gamma, beta = (
                gamma_beta.chunk(
                    2,
                    dim=-1
                )
            )

            z = (
                (
                    1.0
                    +
                    gamma[:, None, :]
                )
                *
                z
                +
                beta[:, None, :]
            )

        # --------------------------------
        # Same Transformer as B3
        # --------------------------------

        if self.fusion == "multifilm":

            # C3: multi-level Physics-FiLM.
            # The physical representation p is shared,
            # while each Transformer depth has its own
            # gamma/beta projection.
            for layer, film_head in zip(
                self.encoder.layers,
                self.film_heads
            ):
                z = layer(z)

                gamma_beta = film_head(p)

                gamma, beta = gamma_beta.chunk(
                    2,
                    dim=-1
                )

                z = (
                    (1.0 + gamma[:, None, :]) * z
                    + beta[:, None, :]
                )

            # Match nn.TransformerEncoder behavior
            # if a final norm is configured.
            if self.encoder.norm is not None:
                z = self.encoder.norm(z)

        else:

            z = self.encoder(
                z
            )

        z = self.norm(
            z.mean(dim=1)
        )

        pred = self.head(
            z
        )

        return pred.view(
            B,
            FUTURE_N,
            3
        )

# ============================================================
# METRICS — B0 compatible
# ============================================================

def safe_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    if len(a) < 2:
        return np.nan

    if (
        np.std(a) < EPS
        or
        np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b
        )[0, 1]
    )


def r2_score_1d(y, pred):

    ss_res = np.sum(
        (y - pred) ** 2
    )

    ss_tot = np.sum(
        (
            y - np.mean(y)
        ) ** 2
    )

    if ss_tot < EPS:
        return np.nan

    return float(
        1.0
        -
        ss_res / ss_tot
    )


def spectral_features(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    xd = (
        x - np.mean(x)
    )

    nperseg = min(
        128,
        len(xd)
    )

    f, pxx = welch(
        xd,
        fs=FS,
        nperseg=nperseg,
        detrend=False,
        scaling="density"
    )

    total = np.sum(
        pxx
    )

    centroid = (
        float(
            np.sum(f * pxx)
            /
            total
        )
        if total > EPS
        else np.nan
    )

    out = {
        "centroid":
            centroid
    }

    for lo, hi, name in BANDS:

        mask = (
            (f >= lo)
            &
            (f < hi)
        )

        if np.sum(mask) >= 2:

            e = float(
                np.trapezoid(
                    pxx[mask],
                    f[mask]
                )
            )

        else:
            e = np.nan

        out[
            f"band_{name}"
        ] = e

    return out


def compute_metrics(y, pred):

    y = np.asarray(
        y,
        dtype=np.float64
    )

    pred = np.asarray(
        pred,
        dtype=np.float64
    )

    err = pred - y

    yd = (
        y - np.mean(y)
    )

    pdyn = (
        pred - np.mean(pred)
    )

    gt_rms = float(
        np.sqrt(
            np.mean(
                yd ** 2
            )
        )
    )

    pr_rms = float(
        np.sqrt(
            np.mean(
                pdyn ** 2
            )
        )
    )

    sy = spectral_features(y)
    sp = spectral_features(pred)

    result = {
        "mae":
            float(
                np.mean(
                    np.abs(err)
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        err ** 2
                    )
                )
            ),

        "corr":
            safe_corr(
                y,
                pred
            ),

        "r2":
            r2_score_1d(
                y,
                pred
            ),

        "dynamic_corr":
            safe_corr(
                yd,
                pdyn
            ),

        "dynamic_r2":
            r2_score_1d(
                yd,
                pdyn
            ),

        "diff_corr":
            safe_corr(
                np.diff(y),
                np.diff(pred)
            ),

        "rms_ratio":
            (
                pr_rms / gt_rms
                if gt_rms > EPS
                else np.nan
            ),

        "centroid_abs_error":
            (
                abs(
                    sp["centroid"]
                    -
                    sy["centroid"]
                )
                if
                np.isfinite(
                    sp["centroid"]
                )
                and
                np.isfinite(
                    sy["centroid"]
                )
                else np.nan
            ),
    }

    for _, _, name in BANDS:

        gt = sy[
            f"band_{name}"
        ]

        pr = sp[
            f"band_{name}"
        ]

        if (
            np.isfinite(gt)
            and
            gt > EPS
            and
            np.isfinite(pr)
        ):

            result[
                f"band_{name}_ratio"
            ] = pr / gt

            result[
                f"band_{name}_log_error"
            ] = abs(
                np.log(
                    (pr + EPS)
                    /
                    (gt + EPS)
                )
            )

        else:

            result[
                f"band_{name}_ratio"
            ] = np.nan

            result[
                f"band_{name}_log_error"
            ] = np.nan

    return result


SUMMARY_METRICS = [
    "mae",
    "rmse",
    "corr",
    "r2",
    "dynamic_corr",
    "dynamic_r2",
    "diff_corr",
    "rms_ratio",
    "centroid_abs_error",
    "band_100_500_ratio",
    "band_100_500_log_error",
    "band_500_1000_ratio",
    "band_500_1000_log_error",
    "band_1000_2000_ratio",
    "band_1000_2000_log_error",
]


def robust_median(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.median(x)
    )


# ============================================================
# FAST VALIDATION FOR CHECKPOINT SELECTION
# ============================================================

@torch.no_grad()
def validate_fast(
    model,
    loader,
    device,
    amp_enabled
):

    model.eval()

    sq = 0.0
    ab = 0.0
    n = 0

    dyn_corrs = []
    dyn_r2s = []
    diff_corrs = []

    for x, y, _ in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled
        ):

            pred = model(x)

        p = (
            pred.detach()
            .float()
            .cpu()
            .numpy()
        )

        t = (
            y.detach()
            .float()
            .cpu()
            .numpy()
        )

        err = p - t

        sq += float(
            np.sum(
                err ** 2
            )
        )

        ab += float(
            np.sum(
                np.abs(err)
            )
        )

        n += int(
            np.prod(
                err.shape
            )
        )

        for bi in range(
            len(p)
        ):

            for ai in range(3):

                yy = t[
                    bi,
                    :,
                    ai
                ]

                pp = p[
                    bi,
                    :,
                    ai
                ]

                yd = yy - yy.mean()
                pd = pp - pp.mean()

                dyn_corrs.append(
                    safe_corr(
                        yd,
                        pd
                    )
                )

                dyn_r2s.append(
                    r2_score_1d(
                        yd,
                        pd
                    )
                )

                diff_corrs.append(
                    safe_corr(
                        np.diff(yy),
                        np.diff(pp)
                    )
                )

    rmse = math.sqrt(
        sq / n
    )

    mae = ab / n

    dyn_corr = robust_median(
        dyn_corrs
    )

    dyn_r2 = robust_median(
        dyn_r2s
    )

    diff_corr = robust_median(
        diff_corrs
    )

    return {
        "rmse_global":
            rmse,

        "mae_global":
            mae,

        "median_dynamic_corr":
            dyn_corr,

        "median_dynamic_r2":
            dyn_r2,

        "median_diff_corr":
            diff_corr,
    }


# ============================================================
# FULL B0-COMPATIBLE FINAL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_full(
    model,
    loader,
    device,
    amp_enabled
):

    model.eval()

    rows = []

    inference_seconds = 0.0

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

        t0 = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled
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

        p = (
            pred.detach()
            .float()
            .cpu()
            .numpy()
        )

        t = y.numpy()

        batch_n = len(p)

        for bi in range(
            batch_n
        ):

            for ai, axis in enumerate(
                AXES
            ):

                m = compute_metrics(
                    t[
                        bi,
                        :,
                        ai
                    ],
                    p[
                        bi,
                        :,
                        ai
                    ]
                )

                out = {
                    "model":
                        "Transformer",

                    "axis":
                        axis,

                    "texture_id":
                        int(
                            meta[
                                "texture_id"
                            ][bi]
                        ),

                    "direction":
                        int(
                            meta[
                                "direction"
                            ][bi]
                        ),

                    "velocity":
                        int(
                            meta[
                                "velocity"
                            ][bi]
                        ),

                    "force_mN":
                        int(
                            meta[
                                "force_mN"
                            ][bi]
                        ),

                    "repeat":
                        int(
                            meta[
                                "repeat"
                            ][bi]
                        ),

                    "window_id":
                        int(
                            meta[
                                "window_id"
                            ][bi]
                        ),

                    "forecast_start":
                        int(
                            meta[
                                "forecast_start"
                            ][bi]
                        ),

                    "window_key_sha256":
                        meta[
                            "window_key_sha256"
                        ][bi],
                }

                out.update(m)

                rows.append(out)

    metrics_df = pd.DataFrame(
        rows
    )

    return (
        metrics_df,
        inference_seconds
    )


def build_summaries(
    metrics_df
):

    axis_rows = []

    for axis, g in metrics_df.groupby(
        "axis"
    ):

        out = {
            "model":
                "Transformer",

            "axis":
                axis,

            "n_windows":
                len(g),
        }

        for metric in SUMMARY_METRICS:

            out[
                f"median_{metric}"
            ] = robust_median(
                g[metric]
            )

        axis_rows.append(out)

    axis_df = pd.DataFrame(
        axis_rows
    )

    overall = {
        "model":
            "Transformer",

        "n_axis_windows":
            len(metrics_df),

        "n_forecast_windows":
            (
                metrics_df[
                    "window_key_sha256"
                ]
                .nunique()
            ),
    }

    for metric in SUMMARY_METRICS:

        overall[
            f"median_{metric}"
        ] = robust_median(
            metrics_df[metric]
        )

    return (
        pd.DataFrame(
            [overall]
        ),
        axis_df
    )


# ============================================================
# MAIN
# ============================================================


def log_magnitude_spectral_loss(
    pred,
    target,
):
    """
    Log-magnitude rFFT loss on future waveform.

    pred/target:
        [B, future_time, 3]

    FFT is performed only along the future-time dimension.
    float32 is used deliberately for numerical stability.
    """

    pred_f = torch.fft.rfft(
        pred.float(),
        dim=1,
        norm="ortho",
    )

    target_f = torch.fft.rfft(
        target.float(),
        dim=1,
        norm="ortho",
    )

    pred_logmag = torch.log1p(
        torch.abs(pred_f)
    )

    target_logmag = torch.log1p(
        torch.abs(target_f)
    )

    return torch.mean(
        torch.abs(
            pred_logmag
            -
            target_logmag
        )
    )


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        default="cuda:0"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--hidden-dim",
        type=int,
        default=128
    )

    p.add_argument(
        "--num-layers",
        type=int,
        default=2
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--patience",
        type=int,
        default=6
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    p.add_argument(
        "--fusion",
        choices=["additive", "film", "multifilm"],
        required=True
    )

    p.add_argument(
        "--condition",
        choices=[
            "velocity",
            "velocity_direction",
            "full"
        ],
        required=True
    )

    p.add_argument(
        "--lambda-spec",
        type=float,
        default=0.05
    )

    p.add_argument(
        "--smoke",
        action="store_true"
    )

    p.add_argument(
        "--protocol-check-only",
        action="store_true"
    )

    return p.parse_args()


def main():

    args = parse_args()

    global ACTIVE_CONDITION_MODE
    ACTIVE_CONDITION_MODE = args.condition

    condition_dim_map = {
        "velocity": 1,
        "velocity_direction": 3,
        "full": 4,
    }

    condition_dim = condition_dim_map[
        args.condition
    ]

    condition_label_map = {
        "velocity":
            "[v_norm]",

        "velocity_direction":
            "[v_norm, sin(theta), cos(theta)]",

        "full":
            "[v_norm, sin(theta), cos(theta), F_norm]",
    }

    condition_label = condition_label_map[
        args.condition
    ]

    seed_everything(
        args.seed
    )

    if (
        args.device.startswith("cuda")
        and
        not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    amp_enabled = (
        device.type == "cuda"
    )

    mode = (
        "smoke"
        if args.smoke
        else "full"
    )

    if args.fusion != "film":
        raise ValueError(
            "Final-Single-FiLM requires --fusion film."
        )

    if args.condition != "full":
        raise ValueError(
            "I8 requires --condition full."
        )

    if args.lambda_spec < 0.0:
        raise ValueError(
            "--lambda-spec must be >= 0."
        )

    lambda_tag = (
        f"{args.lambda_spec:g}"
    )

    out = (
        OUT_ROOT
        / "logmag"
        / f"lambda_{lambda_tag}"
        / f"seed_{args.seed}_{mode}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 78)
    print(
        "Cluster Final — Visual Gated Single-FiLM"
    )
    print("=" * 78)

    print(
        f"Mode          : {mode}"
    )

    print(
        f"Device        : {device}"
    )

    if device.type == "cuda":

        print(
            "GPU           : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Torch         : "
        f"{torch.__version__}"
    )

    print(
        "TEST ACCESSED : NO"
    )

    print(
        f"history       : "
        f"{HISTORY_N} / {HISTORY_MS:g} ms"
    )

    print(
        f"future        : "
        f"{FUTURE_N} / {HORIZON_MS:g} ms"
    )

    print(
        f"stride        : "
        f"{STRIDE_N} / {STRIDE_MS:g} ms"
    )

    manifest = pd.read_csv(
        H4C_MANIFEST
    )

    eval_df = pd.read_csv(
        B0_VAL_INDEX
    )

    if not (
        eval_df["split"]
        .astype(str)
        .str.lower()
        == "val"
    ).all():

        raise RuntimeError(
            "B0 eval index contains "
            "non-VAL rows."
        )

    # Smoke is deliberately tiny.
    train_limit = (
        4096
        if args.smoke
        else None
    )

    val_limit = (
        96
        if args.smoke
        else None
    )

    train_trials, train_windows = (
        build_train_windows(
            manifest,
            max_windows=train_limit
        )
    )

    train_ds = TrainWindowDataset(
        train_trials,
        train_windows
    )

    val_ds = FrozenValDataset(
        eval_df,
        max_windows=val_limit
    )

    print(
        f"TRAIN trials  : "
        f"{len(train_trials)}"
    )

    print(
        f"TRAIN windows : "
        f"{len(train_ds)}"
    )

    print(
        f"VAL windows   : "
        f"{len(val_ds)}"
    )

    sampler = TrialGroupedSampler(
        train_windows,
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False
    )

    model = ConditionedTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patch_len=20,
        nhead=4,
        fusion=args.fusion,
        condition_dim=condition_dim,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Fusion        : "
        f"{args.fusion}"
    )

    print(
        f"Condition     : "
        f"[v_norm, sin(theta), cos(theta), F_norm]"
    )

    print(
        f"Parameters    : "
        f"{n_params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # PyTorch 2.7 AMP API.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    criterion = nn.MSELoss()

    max_epochs = (
        2
        if args.smoke
        else args.epochs
    )

    history_rows = []

    best_score = -np.inf
    best_epoch = -1
    no_improve = 0

    best_path = (
        out
        / "best_frequency.pt"
    )

    training_start = time.time()

    for epoch in range(
        1,
        max_epochs + 1
    ):

        sampler.set_epoch(
            epoch
        )

        model.train()

        running_loss = 0.0
        running_mse = 0.0
        running_spec = 0.0
        seen = 0

        epoch_start = time.time()

        for batch_idx, (
            x,
            y
        ) in enumerate(
            train_loader,
            start=1
        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled
            ):

                pred = model(x)

                mse_loss = criterion(
                    pred,
                    y
                )

                # Spectral loss is computed in float32.
                # rFFT dimension 1 = future-time dimension.
                spec_loss = (
                    log_magnitude_spectral_loss(
                        pred,
                        y
                    )
                )

                loss = (
                    mse_loss
                    +
                    args.lambda_spec
                    *
                    spec_loss
                )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            bs = x.shape[0]

            running_loss += (
                float(
                    loss.detach().item()
                )
                *
                bs
            )

            running_mse += (
                float(
                    mse_loss.detach().item()
                )
                *
                bs
            )

            running_spec += (
                float(
                    spec_loss.detach().item()
                )
                *
                bs
            )

            seen += bs

            if (
                args.smoke
                and
                batch_idx == 1
            ):

                print(
                    "[SMOKE] first batch:"
                )

                print(
                    "  X =",
                    tuple(x.shape)
                )

                print(
                    "  Y =",
                    tuple(y.shape)
                )

                print(
                    "  P =",
                    tuple(pred.shape)
                )

                print(
                    "  mse_loss =",
                    float(
                        mse_loss.detach().item()
                    )
                )

                print(
                    "  spec_loss =",
                    float(
                        spec_loss.detach().item()
                    )
                )

                print(
                    "  total_loss =",
                    float(
                        loss.detach().item()
                    )
                )

        train_total = (
            running_loss
            /
            max(seen, 1)
        )

        train_mse = (
            running_mse
            /
            max(seen, 1)
        )

        train_spec = (
            running_spec
            /
            max(seen, 1)
        )

        val_fast = validate_fast(
            model,
            val_loader,
            device,
            amp_enabled
        )

        # Checkpoint selection deliberately prioritizes
        # dynamic waveform fidelity while requiring RMSE.
        #
        # Score is only for selecting epochs of THIS Transformer;
        # final paper comparison still reports all metrics.
        # Frozen deep-model checkpoint protocol:
        # select the epoch with minimum VAL global RMSE.
        # We keep the variable name "score" so the existing
        # checkpoint/logging code does not need structural changes.
        score = -val_fast[
            "rmse_global"
        ]

        epoch_seconds = (
            time.time()
            -
            epoch_start
        )

        row = {
            "epoch":
                epoch,

            "train_mse":
                train_mse,

            "train_spec":
                train_spec,

            "train_total":
                train_total,

            "lambda_spec":
                args.lambda_spec,

            **val_fast,

            "selection_score":
                score,

            "epoch_seconds":
                epoch_seconds,
        }

        history_rows.append(
            row
        )

        print(
            f"[FINAL-FILM] epoch="
            f"{epoch:02d} | "
            f"train_mse="
            f"{train_mse:.8f} | "
            f"train_spec="
            f"{train_spec:.8f} | "
            f"train_total="
            f"{train_total:.8f} | "
            f"val_rmse="
            f"{val_fast['rmse_global']:.6f} | "
            f"dyn-r="
            f"{val_fast['median_dynamic_corr']:.6f} | "
            f"dyn-R2="
            f"{val_fast['median_dynamic_r2']:.6f} | "
            f"diff-r="
            f"{val_fast['median_diff_corr']:.6f} | "
            f"score="
            f"{score:.6f} | "
            f"time="
            f"{epoch_seconds:.1f}s",
            flush=True
        )

        if (
            np.isfinite(score)
            and
            score > best_score
        ):

            best_score = score
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "args":
                        vars(args),

                    "epoch":
                        epoch,

                    "selection_score":
                        score,

                    "val_fast":
                        val_fast,

                    "protocol":
                        {
                            "fs":
                                FS,

                            "history_n":
                                HISTORY_N,

                            "future_n":
                                FUTURE_N,

                            "stride_n":
                                STRIDE_N,

                            "test_accessed":
                                False,
                        },
                },
                best_path
            )

        else:

            no_improve += 1

        if (
            not args.smoke
            and
            no_improve >= args.patience
        ):

            print(
                f"[I8] Early stopping "
                f"at epoch {epoch}. "
                f"Best epoch={best_epoch}."
            )

            break

    pd.DataFrame(
        history_rows
    ).to_csv(
        out / "training_history.csv",
        index=False
    )

    if not best_path.exists():
        raise RuntimeError(
            "No checkpoint saved."
        )

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    print("")
    print(
        f"[I8] Best epoch = "
        f"{best_epoch}"
    )

    print(
        f"[I8] Best selection score = "
        f"{best_score:.6f}"
    )

    # --------------------------------------------------------
    # Full B0-compatible final evaluation.
    #
    # Smoke: 96 windows.
    # Full: ALL 5664 frozen B0 VAL windows.
    # --------------------------------------------------------

    if args.smoke:

        final_val_ds = (
            FrozenValDataset(
                eval_df,
                max_windows=96
            )
        )

    else:

        final_val_ds = (
            FrozenValDataset(
                eval_df
            )
        )

    final_loader = DataLoader(
        final_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    metrics_df, inference_seconds = (
        evaluate_full(
            model,
            final_loader,
            device,
            amp_enabled
        )
    )

    summary_df, axis_df = (
        build_summaries(
            metrics_df
        )
    )

    metrics_df.to_csv(
        out
        / "C_window_axis_metrics.csv",
        index=False
    )

    summary_df.to_csv(
        out
        / "C_summary.csv",
        index=False
    )

    axis_df.to_csv(
        out
        / "C_axis_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    r = summary_df.iloc[0]

    expected_windows = (
        96
        if args.smoke
        else 5664
    )

    errors = []

    if (
        r[
            "n_forecast_windows"
        ]
        != expected_windows
    ):

        errors.append(
            "Final VAL window count mismatch."
        )

    if len(metrics_df) != (
        expected_windows * 3
    ):

        errors.append(
            "Metric row count mismatch."
        )

    report = []

    report.append(
        "CLUSTER I8 GATED VISUAL RESIDUAL REPORT"
    )

    report.append(
        "=" * 78
    )

    report.append("")

    report.append(
        f"Mode: {mode}"
    )

    report.append(
        f"Condition mode: {args.condition}"
    )

    report.append(
        "TEST ACCESSED: NO"
    )

    report.append(
        f"Seed: {args.seed}"
    )

    report.append(
        f"Device: {device}"
    )

    report.append(
        f"Parameters: {n_params}"
    )

    report.append(
        f"History: "
        f"{HISTORY_MS:g} ms / "
        f"{HISTORY_N}"
    )

    report.append(
        f"Horizon: "
        f"{HORIZON_MS:g} ms / "
        f"{FUTURE_N}"
    )

    report.append(
        f"Stride: "
        f"{STRIDE_MS:g} ms / "
        f"{STRIDE_N}"
    )

    report.append(
        "Loss: MSE + lambda_spec * log-magnitude rFFT L1"
    )

    report.append(
        f"Lambda spec: {args.lambda_spec:g}"
    )

    report.append(
        "FFT axis: future-time"
    )


    report.append(
        f"Physical condition: {condition_label}"
    )

    report.append(
        f"Condition dim: {condition_dim}"
    )

    report.append(
        "Spectral branch: NONE"
    )

    report.append(
        "Spectral loss: log-magnitude rFFT L1"
    )

    report.append("")

    report.append(
        f"Best epoch: {best_epoch}"
    )

    report.append(
        f"Final VAL windows: "
        f"{int(r['n_forecast_windows'])}"
    )

    report.append("")

    report.append(
        "OVERALL ROBUST MEDIAN"
    )

    report.append(
        "-" * 78
    )

    report.append(
        f"Transformer: "
        f"RMSE="
        f"{r['median_rmse']:.6f}, "
        f"MAE="
        f"{r['median_mae']:.6f}, "
        f"r="
        f"{r['median_corr']:.6f}, "
        f"R2="
        f"{r['median_r2']:.6f}, "
        f"dyn-r="
        f"{r['median_dynamic_corr']:.6f}, "
        f"dyn-R2="
        f"{r['median_dynamic_r2']:.6f}, "
        f"diff-r="
        f"{r['median_diff_corr']:.6f}, "
        f"RMS-ratio="
        f"{r['median_rms_ratio']:.6f}, "
        f"centroid-err="
        f"{r['median_centroid_abs_error']:.3f} Hz, "
        f"500-1000-logerr="
        f"{r['median_band_500_1000_log_error']:.6f}"
    )

    report.append("")

    report.append(
        "AXIS RESULTS"
    )

    report.append(
        "-" * 78
    )

    for a in axis_df.itertuples():

        report.append(
            f"{a.axis}: "
            f"RMSE="
            f"{a.median_rmse:.6f}, "
            f"dyn-r="
            f"{a.median_dynamic_corr:.6f}, "
            f"dyn-R2="
            f"{a.median_dynamic_r2:.6f}, "
            f"diff-r="
            f"{a.median_diff_corr:.6f}, "
            f"500-1000-logerr="
            f"{a.median_band_500_1000_log_error:.6f}"
        )

    report.append("")

    report.append(
        "REFERENCE B0 AR(80)"
    )

    report.append(
        "-" * 78
    )

    report.append(
        "RMSE=0.032939"
    )

    report.append(
        "dyn-r=0.711730"
    )

    report.append(
        "dyn-R2=0.448865"
    )

    report.append(
        "diff-r=0.256280"
    )

    report.append(
        "500-1000-logerr=4.090812"
    )

    report.append("")

    report.append(
        "RUNTIME"
    )

    report.append(
        "-" * 78
    )

    report.append(
        f"Training elapsed: "
        f"{(time.time()-training_start)/60:.2f} min"
    )

    report.append(
        f"Final inference: "
        f"{inference_seconds:.4f} s"
    )

    report.append(
        f"Inference ms/window: "
        f"{1000.0*inference_seconds/expected_windows:.6f}"
    )

    report.append("")

    report.append(
        "INTEGRITY"
    )

    report.append(
        "-" * 78
    )

    report.append(
        f"Errors: {len(errors)}"
    )

    report.append(
        f"Metric rows: "
        f"{len(metrics_df)}"
    )

    report.append("")

    if len(errors) == 0:

        if args.smoke:

            report.append(
                "FINAL P DECISION: "
                "PASS_CONDITION_SMOKE"
            )

        else:

            report.append(
                "FINAL P DECISION: "
                "PASS_CONDITION_SCREEN"
            )

    else:

        report.append(
            "FINAL P DECISION: FAIL"
        )

        for e in errors:
            report.append(
                f"ERROR: {e}"
            )

    report_text = "\n".join(
        report
    )

    (
        out
        / "C_decision_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8"
    )

    metadata = {
        "model":
            "PlainTransformer",

        "seed":
            args.seed,

        "test_accessed":
            False,

        "fs":
            FS,

        "history_n":
            HISTORY_N,

        "future_n":
            FUTURE_N,

        "stride_n":
            STRIDE_N,

        "hidden_dim":
            args.hidden_dim,

        "num_layers":
            args.num_layers,

        "dropout":
            args.dropout,

        "batch_size":
            args.batch_size,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "parameters":
            n_params,

        "best_epoch":
            best_epoch,
    }

    with open(
        out / "C_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print("")
    print(report_text)
    print("")


if __name__ == "__main__":
    main()
