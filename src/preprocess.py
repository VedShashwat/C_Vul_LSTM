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


# Synthetic structural vulnerability examples for optional train-only augmentation.
SYNTHETIC_VULNERABLE: List[Tuple[str, int]] = [
    (
        """void process(unsigned int len, char *src) {
    int size = len + 1;
    char *buf = malloc(size);
    memcpy(buf, src, len);
}""",
        1,
    ),
    (
        """void copy_data(char *dst, char *src, int n) {
    int i;
    for (i = 0; i <= n; i++) {
        dst[i] = src[i];
    }
}""",
        1,
    ),
    (
        """void init_buffer(int size) {
    char *buf = malloc(size);
    buf[0] = 0;
    buf[size-1] = 0;
}""",
        1,
    ),
    (
        """void index_data(int idx, char *arr) {
    if (idx < MAX_SIZE) {
        char val = arr[idx];
        process(val);
    }
}""",
        1,
    ),
    (
        """void cleanup(char *p, int err) {
    if (err) {
        free(p);
    }
    free(p);
}""",
        1,
    ),
    (
        """void handle(struct obj *p) {
    free(p);
    if (p->active) {
        p->run();
    }
}""",
        1,
    ),
    (
        """void log_input(char *user_input) {
    printf(user_input);
}""",
        1,
    ),
    (
        """void read_data(int fd, char *buf, int len) {
    read(fd, buf, len);
    process(buf);
}""",
        1,
    ),
    (
        """int recurse(int n) {
    return recurse(n - 1) + n;
}""",
        1,
    ),
    (
        """void shift(char *buf, int offset, int len) {
    char *ptr = buf + offset;
    memmove(ptr, buf, len);
}""",
        1,
    ),
    (
        """void process_items(int count, item_t *items) {
    int i;
    for (i = 0; i < count; i++) {
        if (items[i].type == ADMIN) items[i].level = get_level(i);
    }
}""",
        1,
    ),
    (
        """size_t get_size(short input) {
    return (size_t)input * sizeof(int);
}""",
        1,
    ),
    (
        """void write_val(char *dst, int len, int val) {
    if (len - 1 >= 0) {
        dst[len] = val;
    }
}""",
        1,
    ),
    (
        """int check_bounds(int val) {
    unsigned int uval = val;
    if (uval < MAX) {
        return arr[uval];
    }
    return -1;
}""",
        1,
    ),
    (
        """void update(struct state *s) {
    s->count++;
    if (s->count > LIMIT) {
        free(s->buf);
    }
    use(s->buf);
}""",
        1,
    ),
]


SYNTHETIC_SAFE: List[Tuple[str, int]] = [
    (
        """int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}""",
        0,
    ),
    (
        """int safe_add(int a, int b, int *result) {
    if (b > 0 && a > INT_MAX - b) return -1;
    if (b < 0 && a < INT_MIN - b) return -1;
    *result = a + b;
    return 0;
}""",
        0,
    ),
    (
        """char *safe_copy(const char *src, size_t max_len) {
    size_t len = strnlen(src, max_len);
    char *dst = malloc(len + 1);
    if (!dst) return NULL;
    memcpy(dst, src, len);
    dst[len] = 0;
    return dst;
}""",
        0,
    ),
    (
        """int read_safe(int fd, char *buf, size_t len) {
    ssize_t n = read(fd, buf, len - 1);
    if (n < 0) return -1;
    buf[n] = 0;
    return (int)n;
}""",
        0,
    ),
    (
        """void init(struct obj *p, int val) {
    if (!p) return;
    p->value = val;
    p->ready = 1;
}""",
        0,
    ),
    (
        """int binary_search(int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}""",
        0,
    ),
    (
        """size_t safe_strlen(const char *s, size_t max) {
    size_t i = 0;
    while (i < max && s[i]) i++;
    return i;
}""",
        0,
    ),
    (
        """void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}""",
        0,
    ),
    (
        """int max3(int a, int b, int c) {
    int m = (a > b) ? a : b;
    return (m > c) ? m : c;
}""",
        0,
    ),
    (
        """struct node *alloc_node(int val) {
    struct node *n = calloc(1, sizeof(*n));
    if (!n) return NULL;
    n->val = val;
    n->next = NULL;
    return n;
}""",
        0,
    ),
]


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
    normalized: List[str] = []

    for i, tok in enumerate(tokens):
        if tok in PRESERVE_IDENTIFIERS:
            normalized.append(tok)
            continue

        if IDENTIFIER_PATTERN.match(tok):
            is_func_like = (i + 1) < len(tokens) and tokens[i + 1] == "("
            normalized.append("FUNC_ID" if is_func_like else "VAR_ID")
        else:
            normalized.append(tok)

    return normalized


def preprocess_tokens(code: str) -> List[str]:
    cleaned = clean_code(code)
    tokens = tokenize_code(cleaned)
    tokens = normalize_identifiers(tokens)
    return tokens


def extract_vulnerability_features(code: str) -> List[float]:
    """
    32 features covering both obvious and structural vulnerability patterns.
    """

    f: List[float] = []

    # Group A: dangerous function presence (8).
    for fn in ["gets", "strcpy", "sprintf", "scanf", "memcpy", "alloca", "system", "execve"]:
        f.append(1.0 if re.search(r"\b" + fn + r"\s*\(", code) else 0.0)

    # Group B: structural memory issues (8).
    f.append(1.0 if re.search(r"for\s*\([^)]*;\s*\w+\s*<=\s*\w+", code) else 0.0)

    n_malloc = len(re.findall(r"\b(malloc|calloc|realloc)\s*\(", code))
    n_null_checks = len(re.findall(r"==\s*NULL|!=\s*NULL|if\s*\(\s*!", code))
    f.append(min(max(n_malloc - n_null_checks, 0) / 3.0, 1.0))

    frees = re.findall(r"\bfree\s*\(\s*(\w+)\s*\)", code)
    f.append(1.0 if len(frees) != len(set(frees)) and len(frees) > 1 else 0.0)

    uaf_hit = 0.0
    for var in set(frees):
        if re.search(r"free\s*\(\s*" + re.escape(var) + r"\s*\).*" + re.escape(var) + r"\s*[.\->\[]", code, re.DOTALL):
            uaf_hit = 1.0
            break
    f.append(uaf_hit)

    f.append(1.0 if re.search(r"\w+\s*\+\s*\w+\s*[\[\)]", code) else 0.0)
    f.append(1.0 if re.search(r"malloc\s*\(\s*\w+\s*[+\-*]\s*\w+", code) else 0.0)
    f.append(1.0 if re.search(r"return\s*&\s*\w+", code) else 0.0)

    io_calls = len(re.findall(r"\b(read|write|recv|send|fread|fwrite)\s*\(", code))
    checked_io = len(re.findall(r"(if|while)\s*\([^)]*\b(read|write|recv|send|fread|fwrite)\b", code))
    f.append(1.0 if io_calls > 0 and checked_io == 0 else 0.0)

    # Group C: integer/arithmetic vulnerabilities (6).
    f.append(1.0 if re.search(r"\b(unsigned|size_t|uint)\b[^;]*[<>]=?\s*\b\w+", code) else 0.0)
    f.append(1.0 if re.search(r"(malloc|calloc)\s*\(\s*\w+\s*\*", code) else 0.0)
    f.append(1.0 if re.search(r"\(size_t\)\s*-|\(unsigned\s+int\)\s*-", code) else 0.0)
    f.append(1.0 if re.search(r"malloc\s*\([^)]*-[^)]*\)", code) else 0.0)
    f.append(1.0 if re.search(r"\bshort\b.*malloc|malloc.*\bshort\b", code) else 0.0)
    f.append(1.0 if re.search(r"for\s*\([^)]*;\s*\w+\s*<\s*(len|size|count|n|num)\b", code) else 0.0)

    # Group D: format string / injection (3).
    f.append(1.0 if re.search(r"\b(printf|fprintf|syslog)\s*\(\s*[^\"'%\n]", code) else 0.0)
    f.append(1.0 if re.search(r"\bsprintf\s*\(\s*\w+\s*,", code) else 0.0)
    f.append(1.0 if re.search(r"\bstrcat\s*\(", code) else 0.0)

    # Group E: complexity proxies (4).
    f.append(min(len(code) / 2000.0, 1.0))

    depth = 0
    max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(depth - 1, 0)
    f.append(min(max_depth / 5.0, 1.0))

    deref_count = len(re.findall(r"\*\w+|\w+\s*->", code))
    f.append(min(deref_count / 20.0, 1.0))

    cast_count = len(re.findall(r"\(\s*(int|char|void\s*\*|unsigned|size_t|long)\s*\)", code))
    f.append(min(cast_count / 10.0, 1.0))

    # Group F: control flow (3).
    has_array_access = bool(re.search(r"\w+\[\w+\]", code))
    has_bounds_check = bool(re.search(r"if\s*\([^)]*<\s*(sizeof|MAX|SIZE|LEN|LIMIT|max|size)", code))
    f.append(1.0 if has_array_access and not has_bounds_check else 0.0)

    f.append(1.0 if re.search(r"\bgoto\b", code) else 0.0)

    fn_decl = re.search(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", code)
    if fn_decl:
        fn_name = fn_decl.group(1)
        body = code[fn_decl.end() :]
        is_recursive = bool(re.search(r"\b" + re.escape(fn_name) + r"\s*\(", body))
        has_base_case = bool(re.search(r"if\s*\([^)]*\)\s*(\{\s*return|return)", body[:300]))
        f.append(1.0 if is_recursive and not has_base_case else 0.0)
    else:
        f.append(0.0)

    assert len(f) == 32, f"Feature count mismatch: {len(f)}"
    return f


def augment_train_with_synthetic(
    train_funcs: List[str],
    train_labels: List[int],
    vuln_repeat: int,
    safe_repeat: int,
) -> Tuple[List[str], List[int]]:
    augmented_funcs = list(train_funcs)
    augmented_labels = list(train_labels)

    for code, label in SYNTHETIC_VULNERABLE:
        for _ in range(max(0, vuln_repeat)):
            augmented_funcs.append(code)
            augmented_labels.append(label)

    for code, label in SYNTHETIC_SAFE:
        for _ in range(max(0, safe_repeat)):
            augmented_funcs.append(code)
            augmented_labels.append(label)

    return augmented_funcs, augmented_labels


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

    for token in ["FUNC_ID", "VAR_ID"]:
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
    feature_lists: Sequence[Sequence[float]],
    vocab: Dict[str, int],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    sequences = torch.full((len(token_lists), max_seq_len), vocab[PAD_TOKEN], dtype=torch.long)

    for i, tokens in enumerate(token_lists):
        token_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_seq_len]]
        if token_ids:
            sequences[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

    label_tensor = torch.tensor(labels, dtype=torch.long)
    feature_tensor = torch.tensor(feature_lists, dtype=torch.float32)
    return {"sequences": sequences, "labels": label_tensor, "vuln_features": feature_tensor}


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
    augment_with_synthetic = bool(config["data"].get("augment_with_synthetic", False))
    synthetic_vuln_repeat = int(config["data"].get("synthetic_vuln_repeat", 20))
    synthetic_safe_repeat = int(config["data"].get("synthetic_safe_repeat", 5))

    ensure_dir(processed_dir)

    funcs, labels = load_diversevul_data(raw_dir=raw_dir, seed=seed, target_size=18000)
    split_data = split_dataset(funcs, labels, train_ratio, val_ratio, test_ratio, seed)

    if augment_with_synthetic:
        train_funcs, train_labels = split_data["train"]
        train_funcs, train_labels = augment_train_with_synthetic(
            train_funcs=list(train_funcs),
            train_labels=list(train_labels),
            vuln_repeat=synthetic_vuln_repeat,
            safe_repeat=synthetic_safe_repeat,
        )
        split_data["train"] = (train_funcs, train_labels)
        print(
            "Synthetic augmentation enabled. "
            f"Added {len(SYNTHETIC_VULNERABLE) * synthetic_vuln_repeat} vulnerable and "
            f"{len(SYNTHETIC_SAFE) * synthetic_safe_repeat} safe synthetic samples to train split."
        )

    tokenized = {}
    split_features: Dict[str, List[List[float]]] = {}
    for split_name, (split_funcs, _) in split_data.items():
        split_tokens: List[List[str]] = []
        split_feat_rows: List[List[float]] = []
        for code in tqdm(split_funcs, desc=f"Tokenizing {split_name}", unit="func"):
            split_tokens.append(preprocess_tokens(code))
            split_feat_rows.append(extract_vulnerability_features(code))
        tokenized[split_name] = split_tokens
        split_features[split_name] = split_feat_rows

    vocab = build_vocab(tokenized["train"], max_vocab_size=vocab_size)

    vocab_path = Path(processed_dir) / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=True, indent=2)

    for split_name, (_, split_labels) in split_data.items():
        data_dict = encode_and_pad(
            token_lists=tokenized[split_name],
            labels=split_labels,
            feature_lists=split_features[split_name],
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
