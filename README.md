# LSTM-Based C Vulnerability Detection

This repository contains a modular deep learning pipeline for binary vulnerability detection in C source code using DiverseVul.

Current model suite:

- `lstm` (Vanilla LSTM baseline)
- `bilstm`
- `bilstm_attn` (Bi-LSTM + token attention)
- `cnn_bilstm`
- `bilstm_multihead` (Bi-LSTM + multi-head self-attention)
- `ensemble` (quorum-vote across all 5 models, with inference-time safety veto)

Dataset note:

- The processed split is heavily imbalanced (~94.3% safe, ~5.7% vulnerable), so threshold tuning and weighted sampling are critical.

Tokenizer/vocabulary note:

- The tokenizer was rebuilt to preserve operators/delimiters and critical vulnerability APIs.
- Vocabulary increased from `834` to `10000` tokens.

## Project Structure

```text
DL_Proj/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── attention_viz.py
│   └── utils.py
├── configs/
│   └── config.yaml
├── results/
├── inference.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset Setup

Install Hugging Face datasets:

```bash
pip install datasets
```

Download and save DiverseVul locally:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('claudios/DiverseVul'); ds.save_to_disk('data/raw/diversevul')"
```

Expected columns:

- `func`: C source code string
- `target`: label (`0` safe, `1` vulnerable)

The preprocessing pipeline filters C rows, samples up to ~18,000 examples, and creates stratified train/val/test splits (70/15/15).

## Installation

```bash
pip install -r requirements.txt
```

## Run Pipeline

Preprocess (rebuild vocab and tensors):

```bash
PYTHONPATH=. python src/preprocess.py
```

Train all models:

```bash
python main.py
```

Train only newer models (load baseline checkpoints):

```bash
python main.py --only-new
```

Evaluation only (load checkpoints, tune thresholds on val, evaluate test):

```bash
python main.py --eval-only
```

Benchmark suites (ensemble):

```bash
python run_benchmark.py --suite data/snippets_v2 --model ensemble --save-json
python run_benchmark.py --suite data/snippets_v3 --model ensemble --save-json
```

Demo runner (overall + per-model + v2/v3/v4 suites):

```bash
python demo.py --show-ensemble-failures --save-json
```

This prints a compact table for each suite, an overall aggregate table across suites,
prints the standard test-set model metrics from `results/comparison_table.md` in terminal,
and saves a full JSON artifact to `results/demo_*.json`.

Other options:

```bash
python main.py --skip-preprocess
python main.py --models lstm bilstm bilstm_attn cnn_bilstm bilstm_multihead
```

## Inference

Predict from file:

```bash
python inference.py --model bilstm --file path/to/snippet.c
```

Predict from inline code:

```bash
python inference.py --model ensemble --code "char buf[10]; memcpy(buf, src, strlen(src));"
```

Inference behavior:

- If `--code` is a bare snippet (not a full function), it is automatically wrapped into a demo function body for better context.
- For ensemble inference, a deterministic post-processing safety veto can override borderline false-positive vulnerable predictions when strong safety patterns are detected.

Supported model names:

- `lstm`
- `bilstm`
- `bilstm_attn`
- `cnn_bilstm`
- `bilstm_multihead`
- `ensemble`

For attention model inference, attention maps are saved to:

- `results/inference_attention.png`

## Outputs

Training/evaluation outputs are saved under `results/`:

- Best checkpoints: `{model_name}_best.pt`
- Classification report: `{model_name}_report.txt`
- Confusion matrix: `{model_name}_confusion.png`
- ROC curve: `{model_name}_roc.png`
- Combined curves: `training_curves.png`
- Comparison table: `comparison_table.md`
- Optimal thresholds: `optimal_thresholds.json`
- Ensemble reports: `ensemble_report.txt`, `ensemble_best2_report.txt`, `ensemble_majority_report.txt`
- Benchmark reports: `benchmark_snippets_v2_ensemble_*.json`, `benchmark_snippets_v3_ensemble_*.json`

## Latest Results (Test Set)

Values below come from `results/comparison_table.md` after the latest eval-only run.

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm | 0.68 | 0.7205 | 0.0935 | 0.4452 | 0.1545 | 0.5506 |
| bilstm | 0.65 | 0.7564 | 0.1039 | 0.4258 | 0.1671 | 0.6497 |
| bilstm_attn | 0.61 | 0.5309 | 0.0857 | 0.7419 | 0.1536 | 0.7010 |
| cnn_bilstm | 0.69 | 0.6757 | 0.0999 | 0.5806 | 0.1705 | 0.7044 |
| bilstm_multihead | 0.64 | 0.5772 | 0.0842 | 0.6452 | 0.1490 | 0.6676 |
| ensemble | 0.60 | 0.6746 | 0.1022 | 0.6000 | 0.1746 | 0.6722 |

Ensemble policy snapshot:

- Method: `quorum_vote`
- Quorum: `3`
- Objective: `f2`

## Latest Benchmark Results

Latest benchmark artifacts:

- `results/benchmark_snippets_v2_ensemble_20260411_145850.json`
- `results/benchmark_snippets_v3_ensemble_20260411_145858.json`

Scores:

- v2: `10/10` (TP 5, TN 5, FP 0, FN 0)
- v3: `10/10` (TP 5, TN 5, FP 0, FN 0)

## Report Context

- The vanilla LSTM baseline is intentionally weaker and serves as the comparison anchor.
- Stronger AUC is achieved by hybrid and attention-heavy models (`cnn_bilstm`, `bilstm_multihead`, and `ensemble`).
