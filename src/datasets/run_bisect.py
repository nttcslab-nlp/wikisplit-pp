import random
import re
from pathlib import Path

from tap import Tap

from src.datasets.common import batch_validate_simple, preprocess
from src.utils import save_jsonl, train_val_test_split


class Args(Tap):
    input_dir: Path = "./datasets/bisect/detokenized"
    output_dir: Path = "./datasets/bisect/base"
    seed: int = 42


def main(args: Args):
    random.seed(args.seed)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    data = []
    with (args.input_dir / "train.src").open() as src, (args.input_dir / "train.dst").open() as dst:
        for comp, simp in zip(src, dst):
            comp, simp = preprocess(comp), preprocess(simp)

            diff = max(0, comp.count('"') - simp.count('"'))
            comp = comp.replace('"', "", diff)
            comp = comp.replace("... ", "")

            num_words = len(comp.split())
            quote_ratio = sum(len(x.split()) for x in re.findall(r'"(.*?)"', comp)) / num_words

            if quote_ratio > 0.5:
                comp = comp.replace('"', "")
                simp = simp.replace('"', "")
            data.append({"complex": comp, "simple": simp})

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
