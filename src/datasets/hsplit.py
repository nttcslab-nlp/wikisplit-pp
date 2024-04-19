from pathlib import Path

from tap import Tap

from src.datasets.common import preprocess
from src.utils import save_jsonl, sentence_tokenize


class Args(Tap):
    input_dir: Path = "./datasets/hsplit/detokenized"
    output_dir: Path = "./datasets/hsplit"


def main(args: Args):
    args.output_dir.mkdir(exist_ok=True, parents=True)

    data = []
    comp_lines = (args.input_dir / "complex.txt").read_text().splitlines()
    simp_lines_list = [
        path.read_text().splitlines() for path in args.input_dir.glob("simple-*.txt")
    ]
    for comp, *simps in zip(comp_lines, *simp_lines_list):
        comp = preprocess(comp)
        if not comp:
            continue
        simps = [preprocess(simp) for simp in simps]
        simps = [simp for simp in simps if simp]
        simps = [
            " ".join(s.capitalize().strip() for s in sentence_tokenize(simp)) for simp in simps
        ]
        data.append({"complex": comp, "simples": simps})

    data = [{"id": idx, **x} for idx, x in enumerate(data)]
    save_jsonl(data, args.output_dir / "all.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
