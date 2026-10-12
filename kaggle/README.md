# Kaggle GPU path — chronos-2 fine-tune

GitHub Actions runners have no GPU. `chronos` fine-tunes are the only
recipe where that matters; Kaggle's free tier (~30 GPU-h/week, T4/P100)
covers it. One-time setup:

1. Create a Kaggle account + API key (`kaggle.json` → `~/.kaggle/`).
2. Add the `GITHUB_TOKEN` secret to the kernel (Add-ons → Secrets).
3. `pip install kaggle && kaggle kernels push -p kaggle/` to upload.
4. Schedule it on kaggle.com (kernels support recurring schedules) or
   re-run `kaggle kernels push` to trigger manually.

The notebook clones this repo and runs `tools/train.py --recipe chronos`
(LoRA, bounded `num_steps`), then publishes through the same
release-asset + manifest-update contract as `train.yml`, so the catalog
entry is indistinguishable from a CI-produced one. Public data only —
the hard rule in `tools/pd_datasets.py` applies here too.
