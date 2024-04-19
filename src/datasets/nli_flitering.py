from pathlib import Path

import pandas as pd
from tap import Tap

from src import utils
from src.nli import NLIClassifier


class Args(Tap):
    root_dir: Path = "./datasets"
    dataset_name: str = "wiki-split"

    processor_name: str = "roberta"
    batch_size: int = 256
    max_seq_len: int = 512

    device: str = "cuda:0"
    seed: int = 42


def main(args: Args):
    utils.set_seed(args.seed)
    input_dir = args.root_dir / args.dataset_name / "base"

    nli_classifier = NLIClassifier(
        processor_name=args.processor_name,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    def filter_and_save(
        data: pd.DataFrame,
        method: str,
        split: str,
    ):
        output_dir = args.root_dir / args.dataset_name / args.processor_name / method

        if method == "entailment" or split != "train":
            data = data[data["is_entailment"]]
        elif method == "100k":
            indices = data["entailment_prob"].nlargest(100_000).index
            data = data.iloc[indices]
        elif method == "10k":
            indices = data["entailment_prob"].nlargest(10_000).index
            data = data.iloc[indices]
        elif method == "099":
            indices = data[data["entailment_prob"] >= 0.99].index
            data = data.iloc[indices]

        data = data.sample(frac=1)
        utils.save_jsonl(data, output_dir / f"{split}.jsonl")

    for split in ["train", "val", "test"]:
        data: pd.DataFrame = utils.load_jsonl(input_dir / f"{split}.jsonl")

        results = nli_classifier.split_and_rephrase_nli(
            indices=data["id"].tolist(),
            complex=data["complex"].tolist(),
            simple=data["simple"].tolist(),
        )
        macro_results = results["macro"][["id", "entailment_prob", "is_entailment"]]
        data = data.merge(macro_results, on="id")

        for method in ["entailment", "100k", "10k", "099"]:
            filter_and_save(
                data=data,
                method=method,
                split=split,
            )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
