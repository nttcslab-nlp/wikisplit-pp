from src import utils


def sent_length(text: str) -> int:
    try:
        return len(utils.sentence_tokenize(text))
    except Exception:
        return 1


def calc_under_split(
    complex: list[str],
    generated_simple: list[str],
) -> float:
    inputs = utils.batch_sentence_tokenize(complex)
    outputs = utils.batch_sentence_tokenize(generated_simple)

    count = 0
    for comp, output in zip(inputs, outputs):
        if len(comp) == len(output):
            count += 1
    return count / len(outputs)


def evaluate_statistics(
    indices: list[int],
    complex: list[str],
    simples: list[list[str]],
    generated_simple: list[str],
) -> tuple[dict, list[dict]]:
    avg_sent_num, avg_sent_num_diff = 0, 0

    results = []
    for idx, gen_simp, simps in zip(indices, generated_simple, simples):
        gen_simp_sent_num = sent_length(gen_simp)
        avg_sent_num += gen_simp_sent_num

        total_sent_num_diff = sum(abs(gen_simp_sent_num - sent_length(s)) for s in simps)
        avg_sent_num_diff += total_sent_num_diff / len(simps)

        results.append(
            {
                "id": idx,
                "sent_num": gen_simp_sent_num,
                "sent_num_diff": total_sent_num_diff / len(simps),
                "generated_simple": gen_simp,
                "simples": simps,
            }
        )

    avg_sent_num /= len(generated_simple)
    avg_sent_num_diff /= len(generated_simple)

    metrics = {
        "avg_sent_num": avg_sent_num,
        "avg_sent_num_diff": avg_sent_num_diff,
        "under_split": calc_under_split(complex, generated_simple),
    }
    return metrics, results
