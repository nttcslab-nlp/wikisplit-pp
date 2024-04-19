import gc
import json
import os
import random
from collections.abc import Iterable
from functools import reduce
from inspect import ismethod
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pysbd
import torch
from more_itertools import divide, flatten
from nltk.tokenize import sent_tokenize

seg = pysbd.Segmenter(language="en", clean=False)

T = TypeVar("T")


def train_val_test_split(
    data: Iterable[T],
    k: int = 10,
    do_shuffle: bool = False,
) -> tuple[list[T], list[T], list[T]]:
    if type(data) == pd.DataFrame:
        data: list[T] = data.to_dict("records")
    else:
        data: list[T] = list(data)
    if do_shuffle:
        random.shuffle(data)

    folds = [list(x) for x in divide(k, data)]
    train = list(flatten(folds[: k - 2]))
    val = folds[k - 2]
    test = folds[k - 1]

    return (train, val, test)


def save_jsonl(data: pd.DataFrame | Iterable[dict] | dict[Any, Iterable], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    data.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def save_json(data: dict[Any, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_json(path, lines=True)


def load_json(path: Path | str) -> dict:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return data


def log(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.DataFrame(df.to_dict("records") + [data])
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)


def save_config(data, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data: dict = data.as_dict()
    except Exception:
        try:
            data: dict = data.to_dict()
        except Exception:
            pass

    if not isinstance(data, dict):
        data: dict = vars(data)

    data = {k: v for k, v in data.items() if not ismethod(v)}
    data = {k: v if type(v) in [int, float, bool, None] else str(v) for k, v in data.items()}

    save_json(data, path)


def set_seed(seed: int = None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict_average(dicts: Iterable[dict]) -> dict:
    dicts: list[dict] = list(dicts)
    averaged = {}

    for k, v in dicts[0].items():
        try:
            v = v.item()
        except Exception:
            pass
        if type(v) in [int, float]:
            averaged[k] = v / len(dicts)
        else:
            averaged[k] = [v]

    for d in dicts[1:]:
        for k, v in d.items():
            try:
                v = v.item()
            except Exception:
                pass
            if type(v) in [int, float]:
                averaged[k] += v / len(dicts)
            else:
                averaged[k].append(v)

    return averaged


def torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp16":
        return torch.float16
    else:
        return torch.float32


def merge_dfs(
    dfs: list[pd.DataFrame] | list[dict],
    on: str | list[str] = None,
) -> pd.DataFrame:
    if on is None:
        on = ["id"]
    dfs = [pd.DataFrame(df) for df in dfs]

    if isinstance(on, str):
        on = [on]

    def merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        cols_to_use = on + list(right.columns.difference(left.columns))
        return pd.merge(left, right[cols_to_use], on=on, how="outer")

    return reduce(merge, dfs)


def sentence_tokenize(
    text: str,
    method: str = "pysbd",
) -> list[str]:
    """
    tokenize a text into sentences using pysbd or nltk
    """
    try:
        if method == "pysbd":
            decomposed = list(seg.segment(text))
        elif method == "nltk":
            decomposed = list(sent_tokenize(text)) or [text]
    except Exception:
        decomposed = [text]

    return [d.strip() for d in decomposed]


def _sentence_tokenize(
    texts: list[str],
    method: str = "pysbd",
) -> list[list[str]]:
    return [sentence_tokenize(text=text, method=method) for text in texts]


def batch_sentence_tokenize(
    texts: list[str],
    method: str = "pysbd",
    num_procs: int = os.cpu_count(),
) -> list[list[str]]:
    if len(texts) <= num_procs:
        return [sentence_tokenize(t) for t in texts]

    results: list[list[str]] = []

    with Pool(processes=num_procs) as pool:
        for sentences in pool.starmap(
            _sentence_tokenize,
            zip(divide(num_procs, texts), repeat(method)),
        ):
            results += sentences
    return results


def freeze_params(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def clear_cache() -> None:
    gc.collect()
    # torch.cuda.empty_cache()


def return_on_failure(value: Any = None):
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                return value

        return applicator

    return decorate
