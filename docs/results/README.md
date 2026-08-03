# Compact result artifacts

This directory contains small, reviewable derivatives of the preserved evidence
archives. It deliberately does not contain model weights, CLIP NPZ references,
feature shards, datasets, checkpoints, or multi-megabyte console logs.

Files:

- `submission_results.json`: consolidated cross-phase result snapshot, archive hashes, and claim boundary;
- `phase2_real_clip_alignment.csv`: real-CLIP PyTorch/Jittor alignment for 64/128/192 budgets;
- `phase4b_training_trace.csv`: all 1,344 Projector optimizer steps with raw/rolling NLL, LR, timing, and finite flag;
- `phase4b_validation_curve.csv`: all 13 held-out evaluations;
- `phase5a_kv_cache_summary.csv`: formal cached/uncached correctness, TV, latency, and memory results.
- `clean_readme_walkthrough_04f098d.json`: compact fresh-checkout/fresh-prefix walkthrough result and SHA256 index for external logs.

Regenerate with `scripts/build_submission_assets.py` and the four archived evidence
packages listed in `docs/SUBMISSION_READINESS.md`.
