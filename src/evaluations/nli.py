from src.nli import NLIClassifier


def evaluate_nli(
    nli_classifier: NLIClassifier,
    complex: list[str],
    generated_simple: list[str],
    indices: list[int] = None,
) -> tuple[dict, dict[str, dict]]:
    assert len(complex) == len(generated_simple)

    results = nli_classifier.split_and_rephrase_nli(
        indices=indices,
        complex=complex,
        simple=generated_simple,
    )
    micro_results = results["micro"].rename(columns={"decomposed_simple": "generated_simple"})
    macro_results = results["macro"].rename(columns={"simple": "generated_simple"})

    metrics = {
        "prob_macro_avg": macro_results["entailment_prob"].mean(),
        "prob_micro_avg": micro_results["entailment_prob"].mean(),
        "entailment_rate": sum(macro_results["is_entailment"]) / len(macro_results),
    }

    results = {
        "micro": micro_results.to_dict(orient="records"),
        "macro": macro_results.to_dict(orient="records"),
    }

    return metrics, results
