import random
import shutil
from collections import defaultdict
from itertools import repeat
from pathlib import Path

from more_itertools import flatten
from tap import Tap

from src import utils


class Args(Tap):
    dataset_dir: Path = "./datasets/wiki-split/deberta"
    seed: int = 42
    alpha: int = 10


def main(args: Args):
    random.seed(args.seed)
    input_path = args.dataset_dir / "entailment" / "train.jsonl"
    dataset = utils.load_jsonl(input_path).to_dict("records")
    data = defaultdict(list)

    for example in dataset:
        simple = example["simple"]
        simps = utils.sentence_tokenize(simple)
        data[len(simps)].append(example)

    targets = list(flatten(repeat(data.pop(3) + data.pop(4), args.alpha)))
    dataset = list(flatten(data.values())) + targets

    random.shuffle(dataset)

    utils.save_jsonl(dataset, args.dataset_dir / f"up{args.alpha}" / "train.jsonl")
    shutil.copyfile(
        args.dataset_dir / "entailment" / "val.jsonl",
        args.dataset_dir / f"up{args.alpha}" / "val.jsonl",
    )
    shutil.copyfile(
        args.dataset_dir / "entailment" / "test.jsonl",
        args.dataset_dir / f"up{args.alpha}" / "test.jsonl",
    )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
