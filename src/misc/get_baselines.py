import json
from pathlib import Path

from tap import Tap


class Args(Tap):
    input_dir: Path = "./outputs/system"


def main(args: Args):
    metrics_columns = [
        "corpus_bleu",
        "bert_score_f",
        "bleurt_score",
        "sari",
        "entailment_rate",
        "fkgl",
        "avg_sent_num",
        "under_split",
    ]

    for eval_set in ["hsplit", "wiki-bm", "cont-bm"]:
        print("*" * 80)
        print(eval_set)
        for path in args.input_dir.glob(f"**/openai/**/{eval_set}/metrics.json"):
            print(path)
            with path.open() as f:
                metrics = json.load(f)
            metrics = {kk: vv for v in metrics.values() for kk, vv in v.items()}
            for metric_name in metrics_columns:
                print(" & ", end="")
                if metric_name not in ["corpus_bleu", "fkgl", "avg_sent_num"]:
                    print(f"{metrics[metric_name]*100:.2f}", end="")
                else:
                    print(f"{metrics[metric_name]:.2f}", end="")
            print(" \\\\")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
