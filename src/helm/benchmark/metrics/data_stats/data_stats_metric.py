# Adapted from summ_eval.data_stats_metric (Apache 2.0 / Newsroom fragments).
# Vendored so medhelm[summarization] does not depend on summ-eval (pyemd==0.5.1, torch<2.0).

from collections import Counter
from typing import Any, Dict, List, Optional

import spacy

from helm.benchmark.metrics.data_stats.fragments import Fragments

_en: Optional[Any] = None


def _get_spacy_en():
    global _en
    if _en is None:
        _en = spacy.load("en_core_web_sm")
    return _en


def _find_ngrams(input_list: List[str], n: int):
    return zip(*[input_list[i:] for i in range(n)])


class DataStatsMetric:
    """Extractive summarization statistics (coverage, density, compression)."""

    def __init__(self, n_gram: int = 3, case: bool = False, tokenize: bool = True):
        self.n_gram = n_gram
        self.case = case
        self.tokenize = tokenize

    def evaluate_example(self, summary: str, input_text: str) -> Dict[str, float]:
        if self.tokenize:
            nlp = _get_spacy_en()
            input_text = [tok.text for tok in nlp(input_text, disable=["tagger", "parser", "ner", "textcat"])]
            summary = [tok.text for tok in nlp(summary, disable=["tagger", "parser", "ner", "textcat"])]
        fragments = Fragments(summary, input_text, case=self.case)
        score_dict: Dict[str, float] = {
            "coverage": fragments.coverage(),
            "density": fragments.density(),
            "compression": fragments.compression(),
            "summary_length": float(len(fragments.summary)),
        }
        tokenized_summary = fragments.summary
        tokenized_text = fragments.text
        for i in range(1, self.n_gram + 1):
            input_ngrams = list(_find_ngrams(tokenized_text, i))
            summ_ngrams = list(_find_ngrams(tokenized_summary, i))
            input_ngrams_set = set(input_ngrams)
            summ_ngrams_set = set(summ_ngrams)
            intersect = summ_ngrams_set.intersection(input_ngrams_set)
            try:
                score_dict[f"percentage_novel_{i}-gram"] = (len(summ_ngrams_set) - len(intersect)) / float(
                    len(summ_ngrams_set)
                )
                ngram_counter = Counter(summ_ngrams)
                repeated = [key for key, val in ngram_counter.items() if val > 1]
                score_dict[f"percentage_repeated_{i}-gram_in_summ"] = len(repeated) / float(len(summ_ngrams_set))
            except ZeroDivisionError:
                continue
        return score_dict

    @property
    def supports_multi_ref(self) -> bool:
        return False
