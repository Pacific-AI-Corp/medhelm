# Adapted from https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py
# (used by summ_eval DataStatsMetric; vendored to avoid summ-eval's pyemd/torch pins)

from collections import namedtuple


def normalize(tokens, case: bool = False) -> list[str]:
    """Lowercase tokens unless case preservation is requested."""
    return [str(t).lower() if not case else str(t) for t in tokens]


class Fragments:
    Match = namedtuple("Match", ("summary", "text", "length"))

    def __init__(self, summary, text, case: bool = False):
        if isinstance(summary, str):
            self.summary = summary.split()
        else:
            self.summary = summary
        if isinstance(text, str):
            self.text = text.split()
        else:
            self.text = text

        self._norm_summary = normalize(self.summary, case)
        self._norm_text = normalize(self.text, case)
        self._matches: list[Fragments.Match] = []
        self._match(self._norm_summary, self._norm_text)

    def overlaps(self) -> list[Match]:
        return self._matches

    def coverage(self, summary_base: bool = True) -> float:
        numerator = sum(o.length for o in self.overlaps())
        denominator = len(self.summary) if summary_base else len(self.text)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def density(self, summary_base: bool = True) -> float:
        numerator = sum(o.length**2 for o in self.overlaps())
        denominator = len(self.summary) if summary_base else len(self.text)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def compression(self, text_to_summary: bool = True) -> float:
        ratio = [len(self.text), len(self.summary)]
        try:
            if text_to_summary:
                return ratio[0] / ratio[1]
            return ratio[1] / ratio[0]
        except ZeroDivisionError:
            return 0.0

    def _match(self, a: list[str], b: list[str]) -> None:
        a_start = b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0
            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start
                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1
                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1
            b_start = 0
            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1
