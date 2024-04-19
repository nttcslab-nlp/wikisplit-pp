import datasets as ds
from src.utils import sentence_tokenize


def wikisplit():
    datasets = ds.DatasetDict(
        {
            "all": ds.load_dataset("json", data_files="datasets/wiki-split/base/all.jsonl", split="train"),
            "train": ds.load_dataset("json", data_files="datasets/wiki-split/base/train.jsonl", split="train"),
            "validation": ds.load_dataset("json", data_files="datasets/wiki-split/base/val.jsonl", split="train"),
            "test": ds.load_dataset("json", data_files="datasets/wiki-split/base/test.jsonl", split="train"),
        }
    )
    datasets.push_to_hub("cl-nagoya/wikisplit")


def minwikisplit():
    datasets = ds.DatasetDict(
        {
            "train": ds.load_dataset("json", data_files="datasets/min-wiki-split/base/train.jsonl", split="train"),
            "validation": ds.load_dataset("json", data_files="datasets/min-wiki-split/base/val.jsonl", split="train"),
            "test": ds.load_dataset("json", data_files="datasets/min-wiki-split/base/test.jsonl", split="train"),
            "all": ds.load_dataset("json", data_files="datasets/min-wiki-split/base/all.jsonl", split="train"),
        }
    )
    datasets.push_to_hub("cl-nagoya/min-wikisplit")


def wikisplitpp():
    datasets = ds.DatasetDict(
        {
            "train": ds.load_dataset(
                "json",
                data_files="datasets/wiki-split/deberta/entailment/train.jsonl",
                split="train",
            ),
            "validation": ds.load_dataset(
                "json",
                data_files="datasets/wiki-split/deberta/entailment/val.jsonl",
                split="train",
            ),
            "test": ds.load_dataset(
                "json",
                data_files="datasets/wiki-split/deberta/entailment/test.jsonl",
                split="train",
            ),
        }
    )

    def process(x):
        simple_tokenized: list[str] = sentence_tokenize(x["simple"])
        return {
            "simple_reversed": " ".join(simple_tokenized[::-1]),
            "simple_tokenized": simple_tokenized,
        }

    datasets = datasets.map(process, num_proc=16)
    datasets = datasets.rename_column("simple", "simple_original")
    datasets = datasets.select_columns(
        [
            "id",
            "complex",
            "simple_reversed",
            "simple_tokenized",
            "simple_original",
            "entailment_prob",
            "split",
        ]
    )
    datasets = datasets.sort("id")
    datasets.push_to_hub("cl-nagoya/wikisplit-pp")


def minwikisplitpp():
    datasets = ds.DatasetDict(
        {
            "train": ds.load_dataset(
                "json",
                data_files="datasets/min-wiki-split/deberta/entailment/train.jsonl",
                split="train",
            ),
            "validation": ds.load_dataset(
                "json",
                data_files="datasets/min-wiki-split/deberta/entailment/val.jsonl",
                split="train",
            ),
            "test": ds.load_dataset(
                "json",
                data_files="datasets/min-wiki-split/deberta/entailment/test.jsonl",
                split="train",
            ),
        }
    )

    def process(x):
        simple_tokenized: list[str] = sentence_tokenize(x["simple"])
        return {
            "simple_reversed": " ".join(simple_tokenized[::-1]),
            "simple_tokenized": simple_tokenized,
        }

    datasets = datasets.map(process, num_proc=16)
    datasets = datasets.rename_column("simple", "simple_original")
    datasets = datasets.select_columns(
        [
            "id",
            "complex",
            "simple_reversed",
            "simple_tokenized",
            "simple_original",
            "entailment_prob",
        ]
    )
    datasets = datasets.sort("id")
    datasets.push_to_hub("cl-nagoya/min-wikisplit-pp")


# wikisplit()
# wikisplitpp()
# minwikisplit()
minwikisplitpp()
