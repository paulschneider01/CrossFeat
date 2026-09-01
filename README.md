# CrossFeat

CrossFeat learns to translate local image descriptors between two aligned
modalities. This repository contains the standard public pipeline used to:

1. extract paired RootSIFT descriptors,
2. fit the shared PCA projection,
3. train a modality-conditioned VAE crosser, and
4. evaluate retrieval and matching on held-out pairs.

The public input format is intentionally simple. It covers registered 2D image
pairs and does not prescribe modality-specific preprocessing.

## Installation

Use Python 3.11.

```bash
git clone https://github.com/paulschneider01/CrossFeat.git
cd CrossFeat
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare your modality pair

Give each image pair the same relative path below its modality folder:

```text
my_dataset/
├── train/
│   ├── modality_a/
│   │   ├── case_001.png
│   │   └── group/case_002.png
│   └── modality_b/
│       ├── case_001.png
│       └── group/case_002.png
├── val/
│   ├── modality_a/...
│   └── modality_b/...
└── test/
    ├── modality_a/...
    └── modality_b/...
```

Requirements:

- `train`, `val`, and `test` must all be present and non-empty.
- The two modality folders in a split must contain identical relative paths.
- Paired images must already be spatially aligned and have identical dimensions.
- Supported formats are PNG, JPEG, TIFF, and NumPy `.npy` images.
- Modality names are arbitrary; use the same names in the folders and config.

CrossFeat intentionally does not resize or register public paired-folder inputs.
Dataset-specific normalization, registration, or cropping should be done before
creating this layout.

## Train CrossFeat

Copy [config/config_example.yaml](config/config_example.yaml), then edit only:

```yaml
datasets:
  - name: paired_folders
    data_root: /absolute/path/to/my_dataset
    modalities: [modality_a, modality_b]
    pairs:
      - {source: modality_a, target: modality_b}
```

Use `device: cuda` when training on a CUDA GPU; the example defaults to CPU so
that configuration validation and small runs work on any machine.

Validate and start training:

```bash
python train.py --config config/my_pair.yaml --dry_run
python train.py --config config/my_pair.yaml
```

With the example output settings, artifacts are written to:

```text
experiments/crossfeat/custom_pair/
├── best_model.pt
├── pca.pkl
├── config.json
├── config_source.yaml
└── logs/
```

`best_model.pt`, `pca.pkl`, and `config.json` together are the trained model.
Keep all three in the same directory. Training protects existing model
artifacts from overwrite; change the config `name` for each new run.

## Evaluate the trained model

```bash
python evaluate.py \
  --model_dir experiments/crossfeat/custom_pair \
  --dataset paired_folders \
  --data_root /absolute/path/to/my_dataset \
  --modalities modality_a modality_b \
  --split test \
  --output_json results/my_pair.json
```

Evaluation reports descriptor retrieval accuracy, mutual-nearest-neighbor
matching statistics, and target registration error for the aligned test pairs.
Evaluation chooses CUDA when available and otherwise uses CPU; pass `--device`
only to override that choice. All test cases are evaluated by default; use
`--max_cases N` only for a quick subset check.

## Scope

The recommended public path is deliberately limited to one aligned 2D modality
pair, SIFT/RootSIFT descriptors, shared PCA, and the CrossFeat VAE. Dataset
loaders used for the paper experiments are retained in `src/io`, but new users
do not need them. Unpaired training, automatic registration, and
modality-specific preprocessing recipes are outside this release.

## Repository map

```text
train.py                    small training entrypoint
train_universal.py          descriptor extraction and VAE training
evaluate.py                 standard held-out evaluation entrypoint
evaluate_core.py            matching and metric implementation
config/config_example.yaml  starter config for a custom pair
src/                        model, losses, data loading, and utilities
tests/                      unit and end-to-end public workflow tests
```

## Verify the installation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The end-to-end test creates a tiny aligned dataset, trains for one epoch on CPU,
loads the emitted checkpoint/PCA/config bundle, and evaluates its test split.

## License and citation

The code is released under the [MIT License](LICENSE). If you use CrossFeat in
research, please cite the paper and the software metadata in
[CITATION.cff](CITATION.cff).
