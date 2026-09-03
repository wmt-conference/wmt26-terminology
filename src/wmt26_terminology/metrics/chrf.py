from sacrebleu.metrics import CHRF

from wmt26_terminology.schema import TestSet
from wmt26_terminology.submission import Submission

NAME = "chrf"
_CHRF = CHRF(word_order=2)


def aligned_paragraphs(test_set: TestSet, submission: Submission) -> list[list[tuple[str, str] | None]]:
    """Per document, per paragraph: (reference, hypothesis), None where the paragraph has no
    reference and is excluded on both sides."""
    return [
        [(p.reference, hyp) if p.reference is not None else None for p, hyp in zip(doc.paragraphs, hyp_doc, strict=True)]
        for doc, hyp_doc in zip(test_set.documents, submission.documents, strict=True)
    ]


def document_chrf(test_set: TestSet, submission: Submission) -> float:
    delimiter = test_set.paragraph_delimiter or "\n"
    references, hypotheses = [], []
    for pairs in aligned_paragraphs(test_set, submission):
        kept = [pair for pair in pairs if pair is not None]
        if kept:
            references.append(delimiter.join(ref for ref, _ in kept))
            hypotheses.append(delimiter.join(hyp for _, hyp in kept))
    return _CHRF.corpus_score(hypotheses, [references]).score


def paragraph_chrf(test_set: TestSet, submission: Submission) -> float:
    flat = [pair for pairs in aligned_paragraphs(test_set, submission) for pair in pairs if pair is not None]
    return _CHRF.corpus_score([hyp for _, hyp in flat], [[ref for ref, _ in flat]]).score


def paragraph_scores(test_set: TestSet, submission: Submission) -> list[list[float | None]]:
    return [
        [
            _CHRF.sentence_score(hyp, [ref]).score if pair is not None else None
            for pair in pairs
            for ref, hyp in [pair or ("", "")]
        ]
        for pairs in aligned_paragraphs(test_set, submission)
    ]
