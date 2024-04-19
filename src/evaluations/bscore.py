import bert_score
import pandas as pd


def evaluate_bert_score(
    indices: list[int],
    simples: list[list[str]],
    generated_simple: list[str],
    device: str = None,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict]]:
    P, R, F = bert_score.score(
        generated_simple,
        simples,
        lang="en",
        device=device,
        batch_size=batch_size,
    )

    P_lower, R_lower, F_lower = bert_score.score(
        [s.lower() for s in generated_simple],
        [[s.lower() for s in simp] for simp in simples],
        lang="en",
        device=device,
        batch_size=batch_size,
    )

    metrics = {
        "bert_score_f": float(F.mean()),
        "bert_score_p": float(P.mean()),
        "bert_score_r": float(R.mean()),
        "bert_score_f_lower": float(F_lower.mean()),
        "bert_score_p_lower": float(P_lower.mean()),
        "bert_score_r_lower": float(R_lower.mean()),
    }

    results = pd.DataFrame(
        {
            "id": indices,
            "bert_score_f": F.tolist(),
            "bert_score_p": P.tolist(),
            "bert_score_r": R.tolist(),
            "bert_score_f_lower": F_lower.tolist(),
            "bert_score_p_lower": P_lower.tolist(),
            "bert_score_r_lower": R_lower.tolist(),
            "generated_simple": generated_simple,
            "simples": simples,
        }
    )

    return metrics, results.to_dict(orient="records")
