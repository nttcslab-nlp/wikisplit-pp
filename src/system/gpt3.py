import os
from pathlib import Path

import openai
from tap import Tap
from tqdm import tqdm

from src import utils
from src.evaluations import SplitAndRephraseEvaluator
from src.nli import NLIClassifier

openai.api_key = os.environ.get("OPENAI_API_KEY")


class Args(Tap):
    root_dir: Path = "./datasets"

    model_name: str = "text-davinci-003"
    method: str = "zeroshot"
    nli_processor_name: str = "deberta"
    device: str = "cuda:0"


def main(args: Args):
    output_dir = Path(f"./outputs/system/openai/{args.model_name}/{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    nli_classifier = NLIClassifier(
        processor_name=args.nli_processor_name,
        device=args.device,
    )
    evaluator = SplitAndRephraseEvaluator(
        nli_classifier=nli_classifier,
        device=args.device,
    )

    if args.method == "zeroshot":
        prompt = """
Split and rephrase the following complex sentence into a simple and concise number of sentences, while maintaining the structure, phrases, and meaning of the sentence.

Complex sentence: {comp}
Simple sentences:
            """

    elif args.method == "1shot":
        # taken from the paper of Wiki-Split
        prompt = """
Split and rephrase the following complex sentence into a simple and concise number of sentences, while maintaining the structure, phrases, and meaning of the sentence.

Complex sentence: Street Rod is the first in a series of two games released for the PC and Commodore 64 in 1989.
Simple sentences: Street Rod is the first in a series of two games. It was released for the PC and Commodore 64 in 1989.

Complex sentence: {comp}
Simple sentences:
        """

    elif args.method == "3shot":
        # taken from the paper of Wiki-Split
        prompt = """
Split and rephrase the following complex sentence into a simple and concise number of sentences, while maintaining the structure, phrases, and meaning of the sentence.

Complex sentence: Street Rod is the first in a series of two games released for the PC and Commodore 64 in 1989.
Simple sentences: Street Rod is the first in a series of two games. It was released for the PC and Commodore 64 in 1989.

Complex sentence: He played all 60 minutes in the game and rushed for 114 yards, more yardage than all the Four Horsemen combined.
Simple sentences: He played all 60 minutes in the game. He rushed for 114 yards, more yardage than all the Four Horsemen combined.

Complex sentence: A classic leaf symptom is water-soaked lesions between the veins which appear as angular leaf-spots where the lesion edge and vein meet.
Simple sentences: A classic leaf symptom is the appearance of angular, water-soaked lesions between the veins. The angular appearance results where the lesion edge and vein meet.

Complex sentence: {comp}
Simple sentences:
        """

    elif args.method == "leak":
        # taken from the paper of HSplit, Wiki-BM, Cont-BM
        prompt = """
Split and rephrase the following complex sentence into a simple and concise number of sentences, while maintaining the structure, phrases, and meaning of the sentence.

Complex sentence: One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.
Simple sentences: One side of the armed conflict is composed mainly of the sudanese military and the janjaweed. The latter are a sudanese militia group. They are recruited mostly from afro-arab abbala tribes. These tribes are from the northern rizeigat region in sudan.

Complex sentence: Together with James, she compiled crosswords for several newspapers and magazines, including People, and it was in 1978 that they launched their own publishing company.
Simple sentences: Together with James, she compiled crosswords. It was for several newspapers and magazines, including People. They launched their own publishing company. It was in 1978.

Complex sentence: Except for Supplier's obligations and liability resulting from Section 10.0, Supplier Liability for Third Party Claims, Supplier's liability for any and all claims will be limited to the amount of $1,000,000 USD per occurrence, with an aggregated limit of $4,500,000 USD during the term of this Agreement.
Simple sentences: The following applies, not including the Supplier's obligations and liability resulting from Section 10.0, Supplier Liability for Third Party Claims. Supplier's liability for any and all claims will be limited to the amount of $1,000,000 USD per occurrence. Additionally, there is an aggregated limit of $4,500,000 USD during the term of this Agreement.

Complex sentence: {comp}
Simple sentences:
        """

    def fn(complex: list[str]):
        ret = []
        for comp in tqdm(complex):
            completion = openai.Completion.create(
                model=args.model_name,
                prompt=prompt.format(comp=comp).strip(),
                max_tokens=512,
            )
            generated_simple = completion.choices[0].text.strip()
            ret.append(generated_simple)
        return ret

    metrics, results = evaluator(fn=fn)

    utils.save_json(metrics, output_dir / "all-metrics.json")

    for dataset_name, dataset_results in results.items():
        dir: Path = output_dir / dataset_name
        dir.mkdir(parents=True, exist_ok=True)
        utils.save_json(metrics[dataset_name], dir / "metrics.json")

        dfs = []
        for result_name, result in dataset_results.items():
            utils.save_jsonl(result, dir / f"{result_name}.jsonl")
            if "micro" not in result_name:
                dfs.append(result)

        utils.save_jsonl(utils.merge_dfs(dfs), dir / "results.jsonl")

    utils.save_json(
        {"method": f"openai-{args.method}", "model_name": args.model_name, "dataset_name": None},
        output_dir / "config.json",
    )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
