import sacrebleu
from sacrebleu.metrics.bleu import BLEUScore


# transpose references to match the order of generated_simple
def prepare_for_bleu(references: list[list[str]]) -> list[list[str]]:
    max_refs = max(len(s) for s in references)
    new_references = []
    for simps in references:
        new_references.append(simps + [None] * (max_refs - len(simps)))
    new_references = list(zip(*new_references))
    return new_references


def evaluate_bleu(
    indices: list[int],
    complex: list[str],
    simples: list[list[str]],
    generated_simple: list[str],
) -> tuple[dict, list[dict]]:
    corpus_bleu: BLEUScore = sacrebleu.corpus_bleu(
        hypotheses=generated_simple,
        references=prepare_for_bleu(simples),
        lowercase=True,
    )

    sentence_bleu_score = 0
    results = []
    for idx, comp, simps, gen_simp in zip(
        indices,
        complex,
        simples,
        generated_simple,
    ):
        sentence_bleu = sacrebleu.sentence_bleu(
            hypothesis=gen_simp,
            references=simps,
            lowercase=True,
        )
        sentence_bleu_score += sentence_bleu.score
        results.append(
            {
                "id": idx,
                "sentence_bleu": sentence_bleu.score,
                "sentence_bleu_details": str(sentence_bleu),
                "generated_simple": gen_simp,
                "complex": comp,
                "simples": simps,
            }
        )

    metrics = {
        "corpus_bleu": corpus_bleu.score,
        "corpus_bleu_details": str(corpus_bleu),
        "sentence_bleu": sentence_bleu_score / len(generated_simple),
    }
    return metrics, results
