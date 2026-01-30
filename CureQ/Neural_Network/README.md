# 1D-CNN Training Pipeline (CTRL vs SCA1)

This README describes the full workflow to train the 1D-CNN on MEA-derived features, from raw data → manifests → MEAlytics feature export → electrode filtering → model training.

## Overview of the workflow

1. **Create a data manifest** for your raw recordings (train/val/test selection starts from this list).
2. **Run MEAlytics** to extract features and export CSV files.
3. **Create an electrode-feature manifest** that points to all `Electrode_Features.csv` files.
4. **Run analysis scripts** to generate statistics/plots and an electrode filter table.
5. **Train the 1D-CNN** using the manifests and the electrode filter output.

---

## Step 1 — Create a manifest for the raw data

Use `manifest_maker.py` to scan your data directory and create a manifest CSV with paths to all recordings.

```bash
python manifest_maker.py --data_dir /path/to/data --output_file manifest.csv
```

**Output**
- `manifest.csv` — list of files that will be used to define training/validation/test sets.

---

## Step 2 — Extract features with MEAlytics

Open **MEAlytics** and analyze the MEA recordings. For each recording, export and collect:

- `Electrode_Features.csv`
- `Features.csv`

Put the exported CSVs into clearly separated folders (recommended):

```
features/
  Electrode_Features/
  Features/
```

---

## Step 3 — Create a manifest for `Electrode_Features.csv`

Create a manifest that points to all exported `Electrode_Features.csv` files:

```bash
python SCA1_analyses/manifest/manifest_maker.py   --data_dir /path/to/features/Electrode_Features   --output_file electrode_manifest.csv
```

**Output**
- `electrode_manifest.csv` — list of electrode-feature CSVs used as model input.

---

## Step 4 — Run electrode/spike analyses (for filtering + QC)

Run the analysis pipeline:

1. `SCA1_analyses/main.py`
2. `Analyze_persistent_spikes.py`
3. `Electrode_analyze.py`

Example:

```bash
python SCA1_analyses/main.py --manifest_file electrode_manifest.csv --output_dir /path/to/output
```

These scripts generate multiple plots and summary statistics.

**Important output used for training**
- `activity_analyzed/measurements_spikes_bursts/electrode_spikes_over_measurement.csv`

This file is used during training to **filter electrodes** (e.g., remove low-quality / problematic electrodes based on spike activity).

---

## Step 5 — Train the 1D-CNN

Train the model using your feature manifest (and other required inputs used by your training script).

```bash
python 1D-CNN.py --feature_manifest electrode_manifest.csv --spike_manifest /path/to/...
```

> Note: The second argument name in your command is shown as `--spike_manifest` in the original README. Make sure it matches the actual CLI arguments in your `1D-CNN.py` script.

---

## Outputs

After training, an output directory is created.

### Per round (each training round)
- Best model checkpoint
- Last model checkpoint
- Training/validation **loss** and **accuracy** plots
- CSV with model predictions on the **test set**
- Confusion matrix plot (based on the test set)
- ROC-AUC curve/plot (based on the test set)

### Overall (after all rounds)
- Global best model (best model from the final round)
- Overall confusion matrix and ROC curve
- JSON file listing the train/validation files used
- `Round_Summaries.json` containing the metrics/results of each round
