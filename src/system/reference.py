import random
from pathlib import Path

from tap import Tap

from src import utils
from src.evaluations import SplitAndRephraseEvaluator
from src.nli import NLIClassifier


class Args(Tap):
    dataset_dir: Path = "./datasets"
    nli_processor_name: str = "deberta"
    eval_batch_size: int = 128
    max_seq_len: int = 512

    seed: int = 42
    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"


def main(args: Args):
    utils.set_seed(args.seed)
    output_dir = Path("./outputs/system/reference")

    nli_classifier = NLIClassifier(
        processor_name=args.nli_processor_name,
        device=args.device,
        batch_size=args.eval_batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
    )
    evaluator = SplitAndRephraseEvaluator(
        nli_classifier=nli_classifier,
        batch_size=args.eval_batch_size,
        device=args.device,
    )

    simples_dict = {}
    for dataset_name, dataset in evaluator.datasets.items():
        simples = [random.choice(simps) for simps in dataset.simples]
        simples_dict[dataset_name] = simples

    metrics, results = evaluator.run_with_generated_simple(simples_dict)

    utils.save_json(metrics, output_dir / "all-metrics.json")

    for dataset_name, dataset_results in results.items():
        dir: Path = output_dir / dataset_name
        utils.save_json(metrics[dataset_name], dir / "metrics.json")

        dfs = []
        for result_name, result in dataset_results.items():
            utils.save_jsonl(result, dir / f"{result_name}.jsonl")
            if "micro" not in result_name:
                dfs.append(result)
        utils.save_jsonl(utils.merge_dfs(dfs), dir / "results.jsonl")

    utils.save_json(
        {"method": "reference", "model_name": None, "dataset_name": None},
        output_dir / "config.json",
    )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
