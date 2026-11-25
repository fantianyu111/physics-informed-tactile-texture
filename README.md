# Lightweight tactile texture modeling

Companion repository for *A Study on Lightweight Tactile Texture Modeling with Physical Priors and Adaptive Inference* by Tianyu Fan, Zhangjia Deng, Xincheng Xia, and Dapeng Chen.

The model uses 160 ms of measured three-axis vibration history together with scanning velocity, motion direction, normal force, and texture image features to predict the next 20 ms of vibration at 8 kHz. A Transformer encodes the vibration history. Layer-specific FiLM parameters condition its temporal representations on the physical and visual inputs.

## Release status

This initial revision documents the project. Training and evaluation scripts, environment specifications, and model weights have yet to be added.

## Verified artifact paths

Paths below are relative to the experiment project root. Run the original scripts from that directory when reproducing these paths.

- `build_cluster_H1_leakage_safe_manifest.py` writes `cluster_H1_results/H1_protocol_A_manifest.csv`, the manifest read by the 8 kHz cache builder. The original manifest builder contains a machine-specific output path.
- `build_cluster_swin_multistage_features.py` writes `cluster_visual_features/swin_t_multistage_cropmean_features.npz`.
- `run_FINAL_TEST_visual.py` reads the evaluation-window index at `cluster_FINAL_TEST_results/final_test_eval_index.csv` and loads `cluster_I8_gated_visual_residual_results/logmag/lambda_0.01/seed_42_full/best_frequency.pt`.

The generator `build_final_test_index.py` writes the final evaluation index. The verified experiment index contains 9,440 distinct records from repeat 1 of the test split, with three windows per record (28,320 windows). All 28,320 window keys are unique. The index includes window boundaries and source/cache paths. The paths document the verified experiment layout; the artifacts themselves are not included in this revision.
