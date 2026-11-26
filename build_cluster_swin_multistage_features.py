#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import swin_t, Swin_T_Weights


NUM_TEXTURES = 118
BATCH_SIZE = 64

CROP_ROOT = Path(
    "cluster_haptic_texture_dataset/images/crop_images"
).resolve()

OUT_DIR = Path("cluster_visual_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = (
    OUT_DIR
    / "swin_t_multistage_cropmean_features.npz"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_texture_id(path):
    nums = re.findall(r"\d+", path.name)
    if not nums:
        nums = re.findall(r"\d+", str(path.parent.name))

    for token in nums:
        v = int(token)
        if 0 <= v < NUM_TEXTURES:
            return v

    raise RuntimeError(
        f"Cannot parse texture id from {path}"
    )


class SwinMultiStage(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        feats = []

        for idx, layer in enumerate(
            self.backbone.features
        ):
            x = layer(x)

            # torchvision Swin output:
            # [B, H, W, C]
            if idx in (3, 5, 7):
                feats.append(
                    x.mean(dim=(1, 2))
                )

        if len(feats) != 3:
            raise RuntimeError(
                f"Expected 3 Swin stages, got {len(feats)}"
            )

        # 192 + 384 + 768 = 1344
        return torch.cat(
            feats,
            dim=1
        )


def main():
    if not CROP_ROOT.is_dir():
        raise RuntimeError(
            f"Missing crop root: {CROP_ROOT}"
        )

    image_paths = sorted(
        list(CROP_ROOT.rglob("*.png"))
        + list(CROP_ROOT.rglob("*.jpg"))
        + list(CROP_ROOT.rglob("*.jpeg"))
    )

    if len(image_paths) != 11800:
        raise RuntimeError(
            f"Expected 11800 images, got {len(image_paths)}"
        )

    grouped = {
        i: []
        for i in range(NUM_TEXTURES)
    }

    for p in image_paths:
        tid = parse_texture_id(p)
        grouped[tid].append(p)

    counts = np.asarray(
        [
            len(grouped[i])
            for i in range(NUM_TEXTURES)
        ],
        dtype=np.int64
    )

    if not np.all(counts == 100):
        raise RuntimeError(
            f"Unexpected crop counts: {counts.tolist()}"
        )

    print("Crops/texture:", int(counts[0]))

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    weights = Swin_T_Weights.DEFAULT
    preprocess = weights.transforms()

    backbone = swin_t(
        weights=weights
    )

    model = SwinMultiStage(
        backbone
    ).to(device)

    model.eval()

    features = np.zeros(
        (NUM_TEXTURES, 1344),
        dtype=np.float32
    )

    print("=" * 88)
    print(
        "CLUSTER SWIN-T MULTISTAGE VISUAL FEATURE EXTRACTION"
    )
    print("=" * 88)
    print("Crop root :", CROP_ROOT)
    print("Images    :", len(image_paths))
    print("Textures  :", NUM_TEXTURES)
    print("Counts    :", sorted(set(counts.tolist())))
    print("Device    :", device)

    if torch.cuda.is_available():
        print(
            "GPU       :",
            torch.cuda.get_device_name(0)
        )

    with torch.no_grad():
        for tid in range(NUM_TEXTURES):
            paths = grouped[tid]

            crop_feats = []

            for start in range(
                0,
                len(paths),
                BATCH_SIZE
            ):
                batch_paths = paths[
                    start:start + BATCH_SIZE
                ]

                batch = torch.stack(
                    [
                        preprocess(
                            Image.open(p).convert("RGB")
                        )
                        for p in batch_paths
                    ],
                    dim=0
                ).to(
                    device,
                    non_blocking=True
                )

                feat = model(batch)

                feat = torch.nn.functional.normalize(
                    feat,
                    p=2,
                    dim=1
                )

                crop_feats.append(
                    feat.cpu().numpy()
                )

            crop_feats = np.concatenate(
                crop_feats,
                axis=0
            )

            texture_feat = crop_feats.mean(
                axis=0
            )

            norm = np.linalg.norm(
                texture_feat
            )

            if norm <= 0.0:
                raise RuntimeError(
                    f"Zero norm at texture {tid}"
                )

            texture_feat = (
                texture_feat
                / norm
            )

            features[tid] = texture_feat

            print(
                f"texture={tid:03d} "
                f"crops={len(paths)} "
                f"norm={np.linalg.norm(features[tid]):.6f}"
            )

    if features.shape != (
        NUM_TEXTURES,
        1344
    ):
        raise RuntimeError(
            f"Bad feature shape {features.shape}"
        )

    if not np.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Non-finite visual features."
        )

    np.savez_compressed(
        OUT_FILE,
        features=features,
        texture_ids=np.arange(
            NUM_TEXTURES,
            dtype=np.int64
        ),
        counts=counts,
        crops_per_texture=np.asarray(
            100,
            dtype=np.int64
        ),
        crop_root=np.asarray(
            str(CROP_ROOT)
        ),
        backbone=np.asarray(
            "torchvision_swin_t_"
            "IMAGENET1K_V1_multistage_192_384_768"
        ),
    )

    print()
    print("=" * 88)
    print(
        "VISUAL FEATURE EXTRACTION PASS"
    )
    print("=" * 88)
    print("Shape  :", features.shape)
    print("Saved  :", OUT_FILE)
    print("SHA256 :", sha256_file(OUT_FILE))
    print("TEST ACCESSED : NO")
    print("SENSOR DATA   : NO")


if __name__ == "__main__":
    main()
