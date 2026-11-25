# Lightweight tactile texture modeling

Companion repository for *A Study on Lightweight Tactile Texture Modeling with Physical Priors and Adaptive Inference* by Tianyu Fan, Zhangjia Deng, Xincheng Xia, and Dapeng Chen.

The model uses 160 ms of measured three-axis vibration history together with scanning velocity, motion direction, normal force, and texture image features to predict the next 20 ms of vibration at 8 kHz. A Transformer encodes the vibration history. Layer-specific FiLM parameters condition its temporal representations on the physical and visual inputs.

## Release status

This initial revision documents the project. Training and evaluation scripts, environment specifications, and model weights have yet to be added.
