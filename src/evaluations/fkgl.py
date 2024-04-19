# based on https://github.com/mounicam/BiSECT/blob/40a9c73429f4a2467096974894e748bdf610f342/metrics/fkgl.py

from string import punctuation

from nltk import word_tokenize

from src.evaluations.syllable_counter import SyllableCounter
from src.utils import sentence_tokenize

PUNCTUATION = set(punctuation)


def is_puntuation(word: str):
    return all(ch in PUNCTUATION for ch in word)


def get_fkgl_from_counts(
    word_count: int,
    sentence_count: int,
    syllable_count: int,
) -> tuple[float, float, float]:
    avg_words_per_sent = (1.0 * word_count) / sentence_count
    avg_syll_per_word = (1.0 * syllable_count) / word_count
    return (
        0.39 * avg_words_per_sent + 11.8 * avg_syll_per_word - 15.59,
        avg_words_per_sent,
        avg_syll_per_word,
    )


# all sentences will be convert to lower case
def evaluate_fkgl(texts: list[str]) -> dict[str, float]:
    syllable_counter = SyllableCounter()
    texts = [text.lower() for text in texts]

    def get_counts(text):
        words = word_tokenize(text)
        sentences = sentence_tokenize(text, method="nltk")
        syl_num = sum(syllable_counter.get_feature(w) for w in words)
        return syl_num, len(words), len(sentences)

    words_count = 0
    syllables_count = 0
    sentences_count = 0

    for text in texts:
        try:
            syl_count, w_count, sent_count = get_counts(text)
            syllables_count += syl_count
            sentences_count += sent_count
            words_count += w_count
        except Exception:
            pass

    fkgl, avg_words_per_sent, avg_syll_per_word = get_fkgl_from_counts(words_count, sentences_count, syllables_count)
    return {
        "fkgl": fkgl,
        "avg_words_per_sent": avg_words_per_sent,
        "avg_syll_per_word": avg_syll_per_word,
    }
