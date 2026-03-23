# LSTM-Based C Vulnerability Detection

This project implements a modular deep learning pipeline for binary vulnerability detection in C functions using the DiverseVul dataset.

Implemented models:

- Vanilla LSTM
- Bi-LSTM
- Bi-LSTM + Token Attention

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

## Run Full Pipeline

```bash
python main.py
```

Options:

```bash
python main.py --skip-preprocess
python main.py --models lstm bilstm bilstm_attn
python main.py --models bilstm_attn
```

## Inference

Predict from file:

```bash
python inference.py --model bilstm --file path/to/snippet.c
```

Predict from inline code:

```bash
python inference.py --model bilstm_attn --code "void foo() { char buf[10]; gets(buf); }"
```

Supported model names:

- `lstm`
- `bilstm`
- `bilstm_attn`

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

## Results Table Placeholder

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla LSTM | - | - | - | - | - |
| Bi-LSTM | - | - | - | - | - |
| Bi-LSTM + Attention | - | - | - | - | - |
