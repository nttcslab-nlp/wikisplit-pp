from collections import defaultdict
from pathlib import Path

import pandas as pd
from tap import Tap

from src.datasets.common import preprocess
from src.utils import save_jsonl


class Args(Tap):
    input_dir: Path = "datasets/small-but-mighty/raw"
    cont_dir: Path = "./datasets/cont-bm"
    wiki_dir: Path = "./datasets/wiki-bm"


def main(args: Args):
    args.cont_dir.mkdir(exist_ok=True, parents=True)
    args.wiki_dir.mkdir(exist_ok=True, parents=True)

    cont_df = pd.read_table(args.input_dir / "cont-bm.tsv")
    wiki_df = pd.read_table(args.input_dir / "wiki-bm.tsv")

    cont_bm = defaultdict(list)
    for comp, simp in zip(cont_df["complex"].values, cont_df["simple"].values):
        cont_bm[preprocess(comp)].append(preprocess(simp))
    cont_bm = [
        {"id": idx, "complex": comp, "simples": simps}
        for idx, (comp, simps) in enumerate(cont_bm.items())
    ]
    save_jsonl(cont_bm, args.cont_dir / "all.jsonl")

    wiki_bm = defaultdict(list)
    for comp, simp in zip(wiki_df["complex"].values, wiki_df["simple"].values):
        wiki_bm[preprocess(comp)].append(preprocess(simp))
    wiki_bm = [
        {"id": idx, "complex": comp, "simples": simps}
        for idx, (comp, simps) in enumerate(wiki_bm.items())
    ]
    save_jsonl(wiki_bm, args.wiki_dir / "all.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
