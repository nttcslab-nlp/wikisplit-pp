import pandas as pd
import tensorflow as tf
from bleurt import score

CHECKPOINT = "./data/bleurt"


def evaluate_bleurt(
    indices: list[int],
    simples: list[list[str]],
    generated_simple: list[str],
    device: str = None,
) -> tuple[dict[str, float], list[dict]]:
    # if device is not None:
    #     device_id = int(device.split(":")[-1])
    #     physical_devices = tf.config.list_physical_devices("GPU")
    #     print(physical_devices)
    #     tf.config.set_visible_devices([physical_devices[device_id]], "GPU")

    scorer = score.BleurtScorer(CHECKPOINT)

    original_indices = []
    references_raw, candidates_raw = [], []
    references_lower, candidates_lower = [], []

    for idx, simps, gen_simp in zip(indices, simples, generated_simple):
        original_indices += [idx] * len(simps)

        references_raw += simps
        candidates_raw += [gen_simp for _ in range(len(simps))]

        references_lower += [simp.lower() for simp in simps]
        candidates_lower += [gen_simp.lower() for _ in range(len(simps))]

    micro_scores_raw = scorer.score(references=references_raw, candidates=candidates_raw)
    micro_scores_lower = scorer.score(references=references_lower, candidates=candidates_lower)
    tf.keras.backend.clear_session()

    micro_results = pd.DataFrame(
        {
            "id": original_indices,
            "bleurt_score": micro_scores_raw,
            "bleurt_score_lower": micro_scores_lower,
            "generated_simple": candidates_raw,
            "simples": references_raw,
        }
    )
    bleurt_scores = micro_results.groupby("id")[["bleurt_score", "bleurt_score_lower"]].mean()

    macro_results = pd.DataFrame(
        {
            "id": indices,
            "bleurt_score": bleurt_scores["bleurt_score"].tolist(),
            "bleurt_score_lower": bleurt_scores["bleurt_score_lower"].tolist(),
            "generated_simple": generated_simple,
            "simples": simples,
        }
    )

    metrics = {
        "bleurt_score": bleurt_scores["bleurt_score"].mean(),
        "bleurt_score_lower": bleurt_scores["bleurt_score_lower"].mean(),
    }

    results = {
        "micro": micro_results.to_dict(orient="records"),
        "macro": macro_results.to_dict(orient="records"),
    }

    return metrics, results
