"""
Based on code on HuggingFace
# https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py

due to issues with the original code mentioned here:
# https://github.com/cocoxu/simplification/issues/6
# https://github.com/huggingface/evaluate/issues/376

Original implementation:
# https://github.com/cocoxu/simplification

And another reference implementation:
# https://github.com/mounicam/BiSECT/blob/40a9c73429f4a2467096974894e748bdf610f342/metrics/sari.py
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import sacrebleu
import sacremoses
from packaging import version


@dataclass
class SARIScore:
    keep_f: float
    add_f: float
    delete_p: float
    score: float


def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        # Fix an alleged bug [2] in the keep score computation.
        # keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    #      a target exactly.
    keepscore_precision = 1
    keepscore_recall = 1
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        # Fix an alleged bug [2] in the keep score computation.
        # keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    delscore_precision = 1
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for _addgram in addgramcountergood:
        addtmpscore += 1

    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    addscore_precision = 1
    addscore_recall = 1
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)


def SARIsent(ssent: str, csent: str, rsents: list[str]) -> SARIScore:
    numref = len(rsents)

    s1grams = ssent.split(" ")
    c1grams = csent.split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)

    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

    return SARIScore(
        keep_f=avgkeepscore,
        add_f=avgaddscore,
        delete_p=avgdelscore,
        score=finalscore,
    )


def normalize(
    sentence: str,
    lowercase: bool = True,
    tokenizer: str = "13a",
    return_str: bool = True,
):
    # Normalization is requried for the ASSET dataset (one of the primary
    # datasets in sentence simplification) to allow using space
    # to split the sentence. Even though Wiki-Auto and TURK datasets,
    # do not require normalization, we do it for consistency.
    # Code adapted from the EASSE library [1] written by the authors of the ASSET dataset.
    # [1] https://github.com/feralvam/easse/blob/580bba7e1378fc8289c663f864e0487188fe8067/easse/utils/preprocessing.py#L7

    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ["13a", "intl"]:
        if version.parse(sacrebleu.__version__).major >= 2:
            normalized_sent = sacrebleu.metrics.bleu._get_tokenizer(tokenizer)()(sentence)
        else:
            normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


# all sentences will be convert to lower case
def evaluate_sari(
    indices: list[int],
    complex: list[str],
    simples: list[list[str]],
    generated_simple: list[str],
):
    sari_scores = []
    add_scores = []
    keep_scores = []
    delp_scores = []

    results = []
    for idx, comp, simps, gen_simp in zip(indices, complex, simples, generated_simple):
        sari = SARIsent(
            normalize(gen_simp),
            normalize(comp),
            [normalize(simp) for simp in simps],
        )
        sari_scores.append(sari.score)
        add_scores.append(sari.add_f)
        keep_scores.append(sari.keep_f)
        delp_scores.append(sari.delete_p)

        results.append(
            {
                "id": idx,
                "sari": float(sari.score),
                "add_f": float(sari.add_f),
                "keep_f": float(sari.keep_f),
                "del_p": float(sari.delete_p),
                "generated_simple": gen_simp,
                "complex": comp,
                "simples": simps,
            }
        )

    metrics = {
        "sari": float(np.mean(sari_scores)),
        "add_f": float(np.mean(add_scores)),
        "keep_f": float(np.mean(keep_scores)),
        "del_p": float(np.mean(delp_scores)),
    }
    return metrics, results


if __name__ == "__main__":
    import math

    generated_simple = "One side of the armed conflict is composed mainly of the sudanese military and the janjaweed. The latter are a sudanese militia group. They are recruited mostly from afro-arab abbala tribes. These tribes are from the northern rizeigat region in sudan."
    comp = "One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan."
    simps = [
        "One side of the armed conflict is composed mainly of the sudanese military and the janjaweed. The latter are a sudanese militia group. They are recruited mostly from afro-arab abbala tribes. These tribes are from the northern rizeigat region in sudan.",
        "There is a side of the armed conflicts. It is composed mainly of the sudanese military and the janjaweed. It is a sudanese militia group. It is recruited mostly from the afro-arab abbala tribes. They are from the northern rizeigat region in sudan.",
        "One side of the armed conflicts is composed mainly of the sudanese military and the janjaweed. The latter is a sudanese militia group. It is recruited mostly from the afro-arab abbala tribes of the northern rizeigat region in sudan.",
        "One side of the armed conflicts is composed mainly of the sudanese military and the janjaweed, a sudanese militia group. The janjaweed are recruited mostly from the afro-arab abbala tribes of the northern rizeigat region in sudan.",
    ]
    sari = SARIsent(normalize(generated_simple), normalize(comp), [normalize(simp) for simp in simps])
    print(sari)
    assert math.isclose(sari.score, 0.6469945583350272)
