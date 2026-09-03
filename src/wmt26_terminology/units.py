import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import sentencepiece as spm
from mweralign import align_texts
from pydantic import BaseModel

from wmt26_terminology.models import MT5_TOKENIZER, XLMR_TOKENIZER, fetch
from wmt26_terminology.schema import TestSet
from wmt26_terminology.submission import Submission

# XLM-R encoders (COMET, CometKiwi, XCOMET) truncate every side at max_positions - 2.
XLMR_CAP = 510
# MetricX-24 truncates the whole "source: ... candidate: ... reference: ..." string at 1536.
MT5_CAP = 1536

UnitStatus = Literal["ok", "empty", "over_cap"]


class Unit(BaseModel):
    """One aligned triple. `status` other than ok means the scorer must not call a model:
    an empty hypothesis piece and a piece exceeding an encoder cap both take the worst score."""

    document_index: int
    paragraph_index: int
    segment_index: int
    source: str
    hypothesis: str
    reference: str
    status: UnitStatus
    xlmr_tokens: int
    mt5_tokens: int


@contextmanager
def _silenced() -> Iterator[None]:
    """mweralign's C++ core reports to the process' stdout and stderr."""
    saved = [os.dup(fd) for fd in (1, 2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        for fd, copy in zip((1, 2), saved, strict=True):
            os.dup2(copy, fd)
            os.close(copy)
        os.close(devnull)


def align_hypothesis(references: list[str], hypothesis: str) -> list[str]:
    """Split one hypothesis paragraph into one piece per reference segment, minimizing word
    error rate against the reference segmentation (mweralign). Pieces may be empty."""
    if len(references) == 1 or not hypothesis.strip():
        return [hypothesis.strip(), *[""] * (len(references) - 1)]
    with _silenced():
        aligned = align_texts("\n".join(references), hypothesis)
    pieces = [piece.strip() for piece in aligned.split("\n")]
    if len(pieces) != len(references):
        raise RuntimeError(f"mweralign returned {len(pieces)} pieces for {len(references)} references")
    return pieces


class Budget:
    def __init__(self) -> None:
        self._xlmr = spm.SentencePieceProcessor(model_file=str(fetch(XLMR_TOKENIZER)))
        self._mt5 = spm.SentencePieceProcessor(model_file=str(fetch(MT5_TOKENIZER)))

    def xlmr(self, text: str) -> int:
        return len(self._xlmr.encode(text))

    def mt5(self, text: str) -> int:
        return len(self._mt5.encode(text))

    def unit(self, source: str, hypothesis: str, reference: str, **indices: int) -> Unit:
        xlmr = max(self.xlmr(source), self.xlmr(hypothesis), self.xlmr(reference))
        mt5 = self.mt5(f"source: {source} candidate: {hypothesis} reference: {reference}")
        status: UnitStatus = "ok"
        if not hypothesis.strip():
            status = "empty"
        elif xlmr > XLMR_CAP or mt5 >= MT5_CAP:
            status = "over_cap"
        texts = {"source": source, "hypothesis": hypothesis, "reference": reference}
        return Unit(**texts, **indices, status=status, xlmr_tokens=xlmr, mt5_tokens=mt5)


def segment_units(test_set: TestSet, submission: Submission, budget: Budget) -> list[list[list[Unit]]]:
    """Per document, per paragraph: one unit per gold segment; paragraphs without a reference
    yield an empty list."""
    out = []
    for d, (doc, hyp_doc) in enumerate(zip(test_set.documents, submission.documents, strict=True)):
        doc_units = []
        for p, (paragraph, hypothesis) in enumerate(zip(doc.paragraphs, hyp_doc, strict=True)):
            if paragraph.reference is None or not paragraph.segments:
                doc_units.append([])
                continue
            references = [segment.reference or "" for segment in paragraph.segments]
            pieces = align_hypothesis(references, hypothesis)
            doc_units.append(
                [
                    budget.unit(segment.source, piece, reference, document_index=d, paragraph_index=p, segment_index=s)
                    for s, (segment, reference, piece) in enumerate(zip(paragraph.segments, references, pieces, strict=True))
                ]
            )
        out.append(doc_units)
    return out
