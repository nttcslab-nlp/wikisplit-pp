import json
from pathlib import Path

import pandas as pd
from tap import Tap

from src import utils


class Args(Tap):
    results_dir: Path = "./outputs/results"
    system_dir: Path = "./outputs/system"


def convert(row: dict) -> dict:
    if not isinstance(row["processor_name"], str):
        row["processor_name"] = None

    return row


METRICS_COLUMNS = [
    "corpus_bleu",
    "bert_score_f",
    "bleurt_score",
    "sari",
    "entailment_rate",
    "fkgl",
    "avg_sent_num",
]


def calc_under_split(resutls_path: Path) -> float:
    df = utils.load_jsonl(resutls_path)
    inputs = utils.batch_sentence_tokenize(df["complex"].tolist())
    outputs = utils.batch_sentence_tokenize(df["generated_simple"].tolist())

    count = 0
    for comp, output in zip(inputs, outputs):
        if len(comp) == len(output):
            count += 1
    return count / len(outputs)


def main(args: Args):
    OK = {
        ("split", None): "T5-small & WikiSplit",
        ("split_rec2", None): "T5-small + Rec. & WikiSplit",
        ("split_reverse", "deberta"): "T5-small & WikiSplit++",
        ("split_rec2_reverse", "deberta"): "T5-small + Rec. & WikiSplit++",
    }

    RENAMES = {
        "echo": "Echo",
        "dissim": "DisSim",
        "bisect": "BiSECT Model",
        "reference": "Reference",
    }

    for eval_set in ["hsplit", "wiki-bm", "cont-bm"]:
        print("*" * 80)
        print(eval_set)

        print("\\tabH ", end="")
        for system_name, dataset_name in [
            ("echo", "N/A"),
            ("dissim", "N/A"),
            ("bisect", "BiSECT+WikiSplit"),
            ("reference", "N/A"),
        ]:
            path = args.system_dir / system_name / eval_set / "metrics.json"
            with path.open() as f:
                metrics = json.load(f)
            metrics = {kk: vv for v in metrics.values() for kk, vv in v.items()}

            print(f"{RENAMES[system_name]} & {dataset_name}", end="")
            for metric_name in METRICS_COLUMNS:
                print(" & ", end="")
                if metric_name not in ["corpus_bleu", "fkgl", "avg_sent_num"]:
                    print(f"{metrics[metric_name]*100:.2f}", end="")
                else:
                    print(f"{metrics[metric_name]:.2f}", end="")

            results_path = args.system_dir / system_name / eval_set / "results.jsonl"
            under_split = calc_under_split(results_path) * 100
            print(f" & {under_split:.2f} \\\\")

        print(r"\hline")
        print("\\tabH ", end="")

        for dataset in ["wiki-split"]:
            path = args.results_dir / eval_set / dataset / "all.csv"
            df = pd.read_csv(path)
            results = {}

            for row in df.to_dict(orient="records"):
                row = convert(row)
                if not (row["processor_name"] is None or row["processor_name"] == "deberta"):
                    continue

                if row["count"] < 10:
                    continue

                if (row["method"], row["processor_name"]) not in OK:
                    continue

                results[(row["method"], row["processor_name"])] = row

        for key, identifier in OK.items():
            print(identifier, end="")
            row = results[key]
            for metric_name in METRICS_COLUMNS:
                print(f" & {row[metric_name]:.2f}", end="")
            print(f" & {row['under_split'] * 100:.2f} \\\\")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
