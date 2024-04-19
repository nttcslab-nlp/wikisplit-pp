import random
import re
from pathlib import Path

from tap import Tap

from src.datasets.common import batch_validate_simple, preprocess
from src.utils import save_jsonl, train_val_test_split


class Args(Tap):
    input_path: Path = "./datasets/min-wiki-split/detokenized/all.txt"
    output_dir: Path = "./datasets/min-wiki-split/base"
    seed: int = 42


def validate(comp: str, simp: str) -> bool:
    entities = re.findall(r'"(.*?)"', comp)
    return all(f'"{entity}"' in simp for entity in entities)


def main(args: Args):
    random.seed(args.seed)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Extra spaces is important.
    # Instances that do not follow this format are of low quality.
    SEP = " <#####> "

    data = []

    with args.input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                comp, simp = line.strip().split(SEP)
            except:
                continue
            comp, simp = preprocess(comp), preprocess(simp)

            if validate(comp, simp):
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
