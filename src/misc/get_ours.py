from pathlib import Path

import pandas as pd
from tap import Tap


class Args(Tap):
    input_dir: Path = "./outputs/results"


METRICS_COLUMNS = [
    "corpus_bleu",
    "bert_score_f",
    "bleurt_score",
    "sari",
    "entailment_rate",
    "fkgl",
    "avg_sent_num",
]


def convert(row: dict) -> dict:
    if not isinstance(row["processor_name"], str):
        row["processor_name"] = None

    return row


def method_ablation(args: Args):
    for eval_set in ["hsplit", "wiki-bm", "cont-bm"]:
        RESULTS = [
            ("split", None, " & &"),
            ("split", "deberta", r"\checkmark & &"),
            ("split_reverse", None, r" & \checkmark &"),
            ("split_reverse", "deberta", r"\checkmark & \checkmark &"),
            ("split_rec2", None, r"& & \checkmark"),
            ("split_rec2", "deberta", r"\checkmark & & \checkmark"),
            ("split_rec2_reverse", None, r"& \checkmark & \checkmark"),
            ("split_rec2_reverse", "deberta", r"\checkmark & \checkmark & \checkmark"),
        ]
        results = {}

        print("*" * 80)
        print(eval_set)
        for dataset in ["wiki-split"]:
            path = args.input_dir / eval_set / dataset / "all.csv"
            df = pd.read_csv(path)
            for row in df.to_dict(orient="records"):
                row = convert(row)
                if not (row["processor_name"] is None or row["processor_name"] == "deberta"):
                    continue

                if row["count"] < 10:
                    continue

                results[(row["method"], row["processor_name"])] = row

            for method, processor_name, checkmarks in RESULTS:
                row = results[(method, processor_name)]
                print(checkmarks, end="")
                for metric_name in METRICS_COLUMNS:
                    print(" & ", end="")
                    print(f"{row[metric_name]:.2f}", end="")
                print(f" & {row['under_split'] * 100:.2f} \\\\")


def dataset_ablation(args: Args):
    OK = {
        ("split", None),
        ("split_reverse", "deberta"),
    }

    for eval_set in ["hsplit", "wiki-bm", "cont-bm"]:
        print("*" * 80)
        print(eval_set)
        for dataset in ["wiki-split", "min-wiki-split", "bisect", "all"]:
            print("-" * 80)
            print(dataset)
            path = args.input_dir / eval_set / dataset / "all.csv"
            df = pd.read_csv(path)
            for row in df.to_dict(orient="records"):
                row = convert(row)
                if not (row["processor_name"] is None or row["processor_name"] == "deberta"):
                    continue

                if row["count"] < 10:
                    continue

                if (row["method"], row["processor_name"]) not in OK:
                    continue

                print(row["method"], row["processor_name"])
                for metric_name in METRICS_COLUMNS:
                    print(" & ", end="")
                    print(f"{row[metric_name]:.2f}", end="")
                print(f" & {row['under_split'] * 100:.2f} \\\\")


def main_result(args: Args):
    OK = {
        ("split", None),
        ("split_rec2", None),
        ("split_reverse", "deberta"),
        ("split_rec2_reverse", "deberta"),
    }

    for eval_set in ["hsplit", "wiki-bm", "cont-bm"]:
        print("*" * 80)
        print(eval_set)
        for dataset in ["wiki-split"]:
            path = args.input_dir / eval_set / dataset / "all.csv"
            df = pd.read_csv(path)
            for row in df.to_dict(orient="records"):
                row = convert(row)
                if not (row["processor_name"] is None or row["processor_name"] == "deberta"):
                    continue

                if row["count"] < 10:
                    continue

                if (row["method"], row["processor_name"]) not in OK:
                    continue

                print(row["method"], row["processor_name"])
                for metric_name in METRICS_COLUMNS:
                    print(" & ", end="")
                    print(f"{row[metric_name]:.2f}", end="")
                print(" \\\\")


if __name__ == "__main__":
    args = Args().parse_args()
    # main_result(args)
    dataset_ablation(args)
    # method_ablation(args)
    # nli_ablation(args)
