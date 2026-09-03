import argparse
import time
from pathlib import Path

from wmt26_terminology.metrics import chrf
from wmt26_terminology.metrics.lemma import Lemmatizer
from wmt26_terminology.metrics.terms import score_terms, term_rates
from wmt26_terminology.results import EvaluationResult, ParagraphResult
from wmt26_terminology.schema import TestSet
from wmt26_terminology.submission import TRACK_MODES, Submission, gold_submission, load_submission, parse_filename

METRIC_VERSION = "1"
UNIFIED = Path(__file__).resolve().parents[2] / "data" / "unified"


def load_test_sets(directory: Path = UNIFIED) -> list[TestSet]:
    return [TestSet.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def evaluate(test_set: TestSet, submission: Submission, lemmatizer: Lemmatizer | None) -> EvaluationResult:
    chrf_scores = chrf.paragraph_scores(test_set, submission)
    annotations = score_terms(test_set, submission, lemmatizer)
    paragraphs = [
        ParagraphResult(
            document_index=d,
            paragraph_index=p,
            scored=score is not None,
            chrf=score,
            annotations=annotations[d][p],
        )
        for d, doc_scores in enumerate(chrf_scores)
        for p, score in enumerate(doc_scores)
    ]
    return EvaluationResult(
        system=submission.system,
        mode=submission.mode,
        provider=test_set.provider,
        domain=test_set.domain,
        pair=test_set.pair,
        track=test_set.track,
        metric_version=METRIC_VERSION,
        document_chrf=chrf.document_chrf(test_set, submission),
        paragraph_chrf=chrf.paragraph_chrf(test_set, submission),
        terms=term_rates(annotations, test_set.track),
        paragraphs=paragraphs,
    )


class Lemmatizers:
    def __init__(self, enabled: bool, use_gpu: bool | None) -> None:
        self.enabled, self.use_gpu, self._cache = enabled, use_gpu, {}

    def get(self, lang: str) -> Lemmatizer | None:
        if not self.enabled:
            return None
        if lang not in self._cache:
            self._cache[lang] = Lemmatizer(lang, self.use_gpu)
        return self._cache[lang]


def _format_row(r: EvaluationResult) -> str:
    t = r.terms.get("all")
    terms = "-"
    if t:
        terms = (
            f"{t.annotated:6.1%} {t.inflected_exact:6.1%} {t.annotated_lemma:6.1%} {t.glossary:6.1%} "
            f"{t.glossary_lemma:6.1%} {t.exclusive:6.1%} {t.glossary_lemma_types:6.1%} n={t.occurrences}"
        )
    label = f"{r.pair} t{r.track} {r.domain}"
    return f"{label:<33} {r.system:<12} {r.mode:<7} {r.document_chrf:6.2f} {r.paragraph_chrf:6.2f}  {terms}"


def main() -> None:
    parser = argparse.ArgumentParser(description="score submissions ({system}.{mode}.{domain}.{pair}.json) or the references")
    parser.add_argument("--submissions", type=Path, help="directory with submission files")
    parser.add_argument("--gold", action="store_true", help="score the references against themselves")
    parser.add_argument("--out", type=Path, help="write one result JSON per (system, mode, test set) here")
    parser.add_argument("--track", type=int, choices=[1, 2])
    parser.add_argument("--skip-lemma", action="store_true", help="skip the stanza lemma tiers")
    parser.add_argument("--cpu", action="store_true", help="run stanza on the CPU")
    args = parser.parse_args()
    if args.gold == (args.submissions is not None):
        parser.error("pass exactly one of --gold or --submissions")

    lemmatizers = Lemmatizers(not args.skip_lemma, False if args.cpu else None)
    print(
        f"{'set':<33} {'system':<12} {'mode':<7} {'docCF':>6} {'parCF':>6}  "
        f"{'annot':>6} {'infl':>6} {'a-lem':>6} {'gloss':>6} {'g-lem':>6} {'excl':>6} {'types':>6}"
    )
    for test_set in load_test_sets():
        if args.track and test_set.track != args.track:
            continue
        if args.gold:
            submissions = [gold_submission(test_set)]
        else:
            paths = sorted(
                p for p in args.submissions.glob("*.json") if parse_filename(p.name)[2:] == (test_set.domain, test_set.pair)
            )
            submissions = [
                load_submission(p, test_set) for p in paths if parse_filename(p.name)[1] in TRACK_MODES[test_set.track]
            ]
        has_terms = any(s.terms for d in test_set.documents for p in d.paragraphs for s in p.segments)
        lemmatizer = lemmatizers.get(test_set.target_lang) if has_terms else None
        for submission in submissions:
            started = time.time()
            result = evaluate(test_set, submission, lemmatizer)
            print(_format_row(result) + f"  ({time.time() - started:.0f}s)")
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                name = f"{result.system}.{result.mode}.{result.domain}.{result.pair}.json"
                (args.out / name).write_text(result.model_dump_json(indent=1, exclude_defaults=True), encoding="utf-8")


if __name__ == "__main__":
    main()
