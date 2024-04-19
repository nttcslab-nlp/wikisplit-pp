import random
from pathlib import Path

from tap import Tap

from src.datasets.common import batch_validate_simple, preprocess
from src.utils import save_jsonl, train_val_test_split


class Args(Tap):
    input_dir: Path = "./datasets/wiki-split/detokenized"
    output_dir: Path = "./datasets/wiki-split/base"
    seed: int = 42


def main(args: Args):
    random.seed(args.seed)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    data = []
    # We cannot use WikiSplit test set because it was used for the construction of Wiki-BM.
    for split in ["train", "val", "tune"]:
        lines = (args.input_dir / f"{split}.tsv").read_text().splitlines()
        for line in lines:
            if not line:
                continue
            comp, simp = line.strip().split("\t")
            comp, simp = preprocess(comp), preprocess(simp)

            data.append({"complex": comp, "simple": simp, "split": split})

    is_valid = batch_validate_simple([d["simple"] for d in data])
    data = [d for d, valid in zip(data, is_valid) if valid]

    random.shuffle(data)
    data = [{"id": idx, **x} for idx, x in enumerate(data)]

    save_jsonl(data, args.output_dir / "all.jsonl")

    train, val, test = train_val_test_split(data)
    save_jsonl(train, args.output_dir / "train.jsonl")
    save_jsonl(val, args.output_dir / "val.jsonl")
    save_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
