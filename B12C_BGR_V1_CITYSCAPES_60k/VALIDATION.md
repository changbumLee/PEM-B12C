# Validation (2026-11-02)

- Baseline: pinned `ddce1e5a8bf5d162ce4241c3c166114de1b4b6e3` in `changbumLee/PEM-B12C`.
- GitHub-verified anchors in `pem/config.py` and `pem/modeling/pixel_decoder/pem_pixel_decoder.py`.
- CPU tests 16/16 PASS (zero initialization identity; residual gradient; boundary gate detach; activation warmup; changed feature behavior; shape checks; patch anchor guarding; training CLI contract).
- BGR enabled: B8 supervised boundary logits reused for refinement gating and for original boundary loss, with no GT boundary at inference.
- Zero initialization yields exact residual identity before training. Baseline source code itself is untouched until installer runs.
- Installer uses pinned commit Git blobs for six tracked sources and writes two BGR overlays + add-on module; backups saved before writes. No `git reset`, no commit/index mutation, no user output/checkpoint deletion.
- Smoke test is full batch 8; CUDA OOM, integration behavior, and final 60k mIoU have NOT been tested here.
- Methods are research hypotheses. No claim of mIoU improvement.
