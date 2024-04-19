from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from tap import Tap

from src import utils


class Args(Tap):
    input_dir: Path = "./datasets/wiki-split/intersection/entailment"


def main(args: Args):
    input_path = args.input_dir / "train.jsonl"
    dataset = utils.load_jsonl(input_path).to_dict("records")
    sentences: list[str] = [e["simple"] for e in dataset]
    sentences: list[list[str]] = utils.batch_sentence_tokenize(sentences)
    sent_nums: list[int] = [len(s) for s in sentences]

    output_dir = args.input_dir / "misc"
    utils.save_json(Counter(sent_nums), output_dir / "train_sent_num.json")

    sns.displot(sent_nums, kde=False)
    fig_path = output_dir / "train_sent_num.png"
    plt.savefig(str(fig_path))
    print(f"Saved to {fig_path}")

    for s in sentences:
        if len(s) == 3:
            print(s)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
