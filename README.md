# OffPSL

OffPSL is a minimal offline Pareto set learning implementation. It trains a deterministic preference-to-design model from an offline dataset and does not include experiment sweep code, logs, checkpoints, or dataset files.

## Data

Dataset files are not included in this release. Download the offline-moo datasets separately and place them under:

```text
offline_moo/data/<task_name>/
```

For a task named `zdt2`, the expected files are:

```text
offline_moo/data/zdt2/zdt2-x-0.npy
offline_moo/data/zdt2/zdt2-y-0.npy
```

The original offline-moo dataset download link is:

```text
https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m offpsl.train --task zdt2 --data_dir offline_moo/data --output_dir outputs
```

This writes the learned model, generated designs, generated objective estimates, and training metrics to:

```text
outputs/zdt2/
```

