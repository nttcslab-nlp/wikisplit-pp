from pathlib import Path

from tap import Tap

from src import utils
from src.datasets.common import preprocess
from src.evaluations import SplitAndRephraseEvaluator
from src.nli import NLIClassifier


class Args(Tap):
    data_dir: Path = "./data"
    system_name: str = "dissim"
    system_filename: str = "raw.txt"

    nli_processor_name: str = "deberta"
    eval_batch_size: int = 128
    max_seq_len: int = 512

    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"


def main(args: Args):
    output_dir = Path(f"./outputs/system/{args.system_name}")
    data_dir = args.data_dir / args.system_name

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

    generated_simples_dict = {}
    for dataset_name in evaluator.datasets.keys():
        path = data_dir / dataset_name / args.system_filename
        generated_simples = path.read_text().strip().splitlines()
        generated_simples = [preprocess(s) for s in generated_simples]
        generated_simples_dict[dataset_name] = generated_simples

    print({k: len(v) for k, v in generated_simples_dict.items()})

    metrics, results = evaluator.run_with_generated_simple(generated_simples_dict)

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
        {"method": args.system_name, "model_name": None, "dataset_name": None},
        output_dir / "config.json",
    )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
