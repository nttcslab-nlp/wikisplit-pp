from collections.abc import Callable
from dataclasses import dataclass

from src import utils
from src.evaluations.bleu import evaluate_bleu
from src.evaluations.bleurt_score import evaluate_bleurt
from src.evaluations.bscore import evaluate_bert_score
from src.evaluations.fkgl import evaluate_fkgl
from src.evaluations.nli import evaluate_nli
from src.evaluations.sari import evaluate_sari
from src.evaluations.statistics import evaluate_statistics
from src.nli import NLIClassifier


@dataclass
class EvaluationDataset:
    indices: list[int]
    complex: list[str]
    simples: list[list[str]]


class SplitAndRephraseEvaluator:
    def __init__(
        self,
        nli_classifier: NLIClassifier,
        device: str,
        batch_size: int = 32,
    ) -> None:
        self.nli_classifier = nli_classifier
        self.device = device
        self.batch_size = batch_size

        self.datasets = {
            "hsplit": self.load_dataset("datasets/hsplit/all.jsonl"),
            "wiki-bm": self.load_dataset("datasets/wiki-bm/all.jsonl"),
            "cont-bm": self.load_dataset("datasets/cont-bm/all.jsonl"),
        }

    def load_dataset(self, dataset_name: str) -> EvaluationDataset:
        dataset = utils.load_jsonl(dataset_name).to_dict("records")
        return EvaluationDataset(
            indices=[d["id"] for d in dataset],
            complex=[d["complex"] for d in dataset],
            simples=[d["simples"] for d in dataset],
        )

    def __call__(
        self,
        fn: Callable[[list[str]], list[str]],
    ) -> tuple[dict[str, dict], dict[str, list[dict]]]:
        metrics, results = {}, {}
        for k, v in self.datasets.items():
            metrics[k], results[k] = self.run(fn, v)
        return metrics, results

    def run_with_generated_simple(self, generated_simples_dict: dict[str, list[str]]):
        metrics, results = {}, {}
        for k in self.datasets.keys():
            generated_simples = generated_simples_dict[k]
            eval_dataset = self.datasets[k]
            metrics[k], results[k] = self.run(lambda _: generated_simples, eval_dataset)
        return metrics, results

    def run(
        self,
        fn: Callable[[list[str]], list[str]],
        dataset: EvaluationDataset,
    ) -> tuple[dict, list[dict]]:
        utils.clear_cache()
        generated_simple = fn(dataset.complex)

        stat_metrics, stat_results = evaluate_statistics(
            indices=dataset.indices,
            complex=dataset.complex,
            simples=dataset.simples,
            generated_simple=generated_simple,
        )

        bleurt_metrics, bleurt_results = evaluate_bleurt(
            indices=dataset.indices,
            simples=dataset.simples,
            generated_simple=generated_simple,
            device=self.device,
        )

        bert_score_metrics, bert_score_results = evaluate_bert_score(
            indices=dataset.indices,
            simples=dataset.simples,
            generated_simple=generated_simple,
            device=self.device,
            batch_size=self.batch_size,
        )

        sari_metrics, sari_results = evaluate_sari(
            indices=dataset.indices,
            complex=dataset.complex,
            simples=dataset.simples,
            generated_simple=generated_simple,
        )

        bleu_metrics, bleu_results = evaluate_bleu(
            indices=dataset.indices,
            complex=dataset.complex,
            simples=dataset.simples,
            generated_simple=generated_simple,
        )

        fkgl_metrics = evaluate_fkgl(generated_simple)

        nli_metrics, nli_results = evaluate_nli(
            nli_classifier=self.nli_classifier,
            indices=dataset.indices,
            complex=dataset.complex,
            generated_simple=generated_simple,
        )

        metrics = {
            "nli": nli_metrics,
            "bleu": bleu_metrics,
            "bert_score": bert_score_metrics,
            "bleurt": bleurt_metrics,
            "sari": sari_metrics,
            "fkgl": fkgl_metrics,
            "statistics": stat_metrics,
        }

        results = {
            "nli_macro": nli_results["macro"],
            "nli_micro": nli_results["micro"],
            "bleu": bleu_results,
            "bert_score": bert_score_results,
            "bleurt_macro": bleurt_results["macro"],
            "bleurt_micro": bleurt_results["micro"],
            "sari": sari_results,
            "statistics": stat_results,
        }

        return metrics, results
