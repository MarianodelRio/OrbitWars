# GPU Setup Guide

## Prerequisites

- GCP VM with GPU (T4 or better)
- NVIDIA driver installed on the VM
- Python 3.10+ on the VM
- `gcloud` CLI configured locally with `--project` and `--zone`

## Setup in 3 commands

```bash
git clone <repo-url> && cd OrbitWars
bash setup.sh
source .venv/bin/activate
```

## Transfer data from Windows

**Option A (primary):**

```bash
gcloud compute scp --recurse ./data/ <VM_NAME>:~/OrbitWars/data/ --project <PROJECT> --zone <ZONE>
```

**Option B:** requires SSH key or WSL/Git Bash

```bash
rsync -avz ./data/ <user>@<VM_IP>:~/OrbitWars/data/
```

## Training commands

```bash
make train                                      # IL training (flat MLP)
make train-rl CONFIG=training/rl_config.json   # RL training
make test-unit                                  # run unit tests
```

## Note on rl_config.json

`training/rl_config.json` is **not** committed to the repo. `make train-rl` will fail with a clear error if the file is missing. You must create or copy this file before running RL training.

## Troubleshooting

- **`nvidia-smi: command not found`** — NVIDIA driver is not installed; `setup.sh` falls back to the CPU wheel automatically.
- **`torch.cuda.is_available()` returns `False` on a GPU VM** — check your driver version; re-run `setup.sh` (it is idempotent).
- **`make train-rl` fails with "No such file"** — `training/rl_config.json` is missing; see the note above.
