import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import ensure_dir, load_config, seed_everything

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
STRING_TOKEN = "STRING_LIT"
NUMBER_TOKEN = "NUM_LIT"

_TOKEN_PATTERN = re.compile(
    r"(?P<COMMENT>//[^\n]*|/\*.*?\*/)"
    r"|(?P<STRING>\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')"
    r"|(?P<NUMBER>\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[uUlLfF]*\b)"
    r"|(?P<MULTIOP>->|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%&|^]=|\+\+|--)"
    r"|(?P<IDENT>[a-zA-Z_]\w*)"
    r"|(?P<SINGLEOP>[{}()\[\];,.<>=+\-*/%&|^~!?:])",
    re.DOTALL,
)

PRESERVE_EXACT = {
    "if",
    "else",
    "while",
    "for",
    "do",
    "return",
    "break",
    "continue",
    "switch",
    "case",
    "default",
    "goto",
    "typedef",
    "struct",
    "union",
    "enum",
    "sizeof",
    "typeof",
    "static",
    "extern",
    "const",
    "volatile",
    "register",
    "auto",
    "inline",
    "int",
    "char",
    "void",
    "short",
    "long",
    "float",
    "double",
    "unsigned",
    "signed",
    "gets",
    "fgets",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "strcmp",
    "strncmp",
    "sprintf",
    "snprintf",
    "vsprintf",
    "vsnprintf",
    "printf",
    "fprintf",
    "scanf",
    "sscanf",
    "fscanf",
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "alloca",
    "valloc",
    "memalign",
    "posix_memalign",
    "read",
    "write",
    "recv",
    "recvfrom",
    "recvmsg",
    "send",
    "sendto",
    "getenv",
    "putenv",
    "system",
    "popen",
    "execve",
    "execl",
    "execlp",
    "open",
    "fopen",
    "fclose",
    "fread",
    "fwrite",
    "fputs",
    "strlen",
    "strdup",
    "strtok",
    "atoi",
    "atol",
    "atof",
    "strtol",
    "strtoul",
    "assert",
    "abort",
    "exit",
    "_exit",
    "longjmp",
    "setjmp",
    "mmap",
    "munmap",
    "mprotect",
    "ioctl",
    "fcntl",
    "new",
    "delete",
    "NULL",
    "nullptr",
    "EOF",
    "stdin",
    "stdout",
    "stderr",
    STRING_TOKEN,
    NUMBER_TOKEN,
}

PRESERVE_IDENTIFIERS = set(PRESERVE_EXACT)
MIN_FREQ = 2

COMMENT_PATTERN = re.compile(r"//.*?$|/\*.*?\*/", flags=re.MULTILINE | re.DOTALL)
STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
WHITESPACE_PATTERN = re.compile(r"\s+")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def clean_code(code: str) -> str:
    code = COMMENT_PATTERN.sub(" ", code)
    code = STRING_PATTERN.sub(f" {STRING_TOKEN} ", code)
    code = NUMBER_PATTERN.sub(f" {NUMBER_TOKEN} ", code)
    code = WHITESPACE_PATTERN.sub(" ", code).strip()
    return code


def tokenize_code(code: str) -> List[str]:
    tokens: List[str] = []

    for match in _TOKEN_PATTERN.finditer(code):
        kind = match.lastgroup
        val = match.group()

        if kind == "COMMENT":
            continue
        if kind == "STRING":
            tokens.append(STRING_TOKEN)
            continue
        if kind == "NUMBER":
            tokens.append(NUMBER_TOKEN)
            continue
        if kind == "MULTIOP" or kind == "SINGLEOP":
            tokens.append(val)
            continue
        if kind == "IDENT":
            if val in PRESERVE_EXACT:
                tokens.append(val)
            else:
                tokens.append(val)

    return tokens


def normalize_identifiers(tokens: Sequence[str]) -> List[str]:
    var_map: Dict[str, str] = {}
    func_map: Dict[str, str] = {}
    normalized: List[str] = []

    for i, tok in enumerate(tokens):
        if tok in PRESERVE_IDENTIFIERS:
            normalized.append(tok)
            continue

        if IDENTIFIER_PATTERN.match(tok):
            is_func_like = (i + 1) < len(tokens) and tokens[i + 1] == "("
            if is_func_like:
                if tok not in func_map:
                    func_map[tok] = f"FUNC_{len(func_map)}"
                normalized.append(func_map[tok])
            else:
                if tok not in var_map:
                    var_map[tok] = f"VAR_{len(var_map)}"
                normalized.append(var_map[tok])
        else:
            normalized.append(tok)

    return normalized


def preprocess_tokens(code: str) -> List[str]:
    cleaned = clean_code(code)
    tokens = tokenize_code(cleaned)
    return tokens


def build_vocab(token_lists: Sequence[Sequence[str]], max_vocab_size: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for token in sorted(PRESERVE_EXACT):
        if len(vocab) >= max_vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    for token, count in counter.most_common(max_vocab_size - 2):
        if count < MIN_FREQ:
            continue
        if len(vocab) >= max_vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_and_pad(
    token_lists: Sequence[Sequence[str]],
    labels: Sequence[int],
    vocab: Dict[str, int],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    sequences = torch.full((len(token_lists), max_seq_len), vocab[PAD_TOKEN], dtype=torch.long)

    for i, tokens in enumerate(token_lists):
        token_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_seq_len]]
        if token_ids:
            sequences[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

    label_tensor = torch.tensor(labels, dtype=torch.long)
    return {"sequences": sequences, "labels": label_tensor}


def _to_list(dataset, key: str) -> List:
    return [row[key] for row in dataset]


def load_diversevul_data(raw_dir: str, seed: int, target_size: int = 18000) -> Tuple[List[str], List[int]]:
    ds = load_from_disk(raw_dir)

    if isinstance(ds, DatasetDict):
        merged = concatenate_datasets([ds[split] for split in ds.keys()])
    else:
        merged = ds

    column_names = set(merged.column_names)
    if "func" not in column_names or "target" not in column_names:
        raise ValueError("DiverseVul dataset must contain 'func' and 'target' columns.")

    lang_column = None
    for candidate in ["lang", "language", "programming_language"]:
        if candidate in column_names:
            lang_column = candidate
            break

    if lang_column is not None:
        merged = merged.filter(lambda x: str(x[lang_column]).strip().lower() == "c")

    funcs = _to_list(merged, "func")
    labels = _to_list(merged, "target")

    cleaned_funcs: List[str] = []
    cleaned_labels: List[int] = []
    for func, label in zip(funcs, labels):
        if isinstance(func, str) and str(label) in {"0", "1"}:
            cleaned_funcs.append(func)
            cleaned_labels.append(int(label))

    if len(cleaned_funcs) == 0:
        raise ValueError("No valid C-language samples found in the dataset.")

    if len(cleaned_funcs) > target_size:
        all_indices = list(range(len(cleaned_funcs)))
        sampled_indices, _ = train_test_split(
            all_indices,
            train_size=target_size,
            stratify=cleaned_labels,
            random_state=seed,
        )
        cleaned_funcs = [cleaned_funcs[i] for i in sampled_indices]
        cleaned_labels = [cleaned_labels[i] for i in sampled_indices]

    return cleaned_funcs, cleaned_labels


def split_dataset(
    funcs: Sequence[str],
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_funcs, temp_funcs, train_labels, temp_labels = train_test_split(
        list(funcs),
        list(labels),
        test_size=(1.0 - train_ratio),
        stratify=list(labels),
        random_state=seed,
    )

    val_portion_of_temp = val_ratio / (val_ratio + test_ratio)
    val_funcs, test_funcs, val_labels, test_labels = train_test_split(
        temp_funcs,
        temp_labels,
        test_size=(1.0 - val_portion_of_temp),
        stratify=temp_labels,
        random_state=seed,
    )

    return {
        "train": (train_funcs, train_labels),
        "val": (val_funcs, val_labels),
        "test": (test_funcs, test_labels),
    }


def preprocess_and_save(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    seed = int(config["training"]["seed"])
    seed_everything(seed)

    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    max_seq_len = int(config["data"]["max_seq_len"])
    vocab_size = int(config["data"]["vocab_size"])
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"]["test_ratio"])

    ensure_dir(processed_dir)

    funcs, labels = load_diversevul_data(raw_dir=raw_dir, seed=seed, target_size=18000)
    split_data = split_dataset(funcs, labels, train_ratio, val_ratio, test_ratio, seed)

    tokenized = {}
    for split_name, (split_funcs, _) in split_data.items():
        tokenized[split_name] = [
            preprocess_tokens(code)
            for code in tqdm(split_funcs, desc=f"Tokenizing {split_name}", unit="func")
        ]

    vocab = build_vocab(tokenized["train"], max_vocab_size=vocab_size)

    vocab_path = Path(processed_dir) / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=True, indent=2)

    for split_name, (_, split_labels) in split_data.items():
        data_dict = encode_and_pad(
            token_lists=tokenized[split_name],
            labels=split_labels,
            vocab=vocab,
            max_seq_len=max_seq_len,
        )
        out_path = Path(processed_dir) / f"{split_name}_data.pt"
        torch.save(data_dict, out_path)


def preprocess_single(code_str: str, vocab: Dict[str, int], max_seq_len: int = 500) -> torch.Tensor:
    tokens = preprocess_tokens(code_str)
    token_ids = [vocab.get(tok, vocab.get(UNK_TOKEN, 1)) for tok in tokens[:max_seq_len]]

    tensor = torch.full((1, max_seq_len), vocab.get(PAD_TOKEN, 0), dtype=torch.long)
    if token_ids:
        tensor[0, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DiverseVul for LSTM training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_and_save(args.config)


if __name__ == "__main__":
    main()
