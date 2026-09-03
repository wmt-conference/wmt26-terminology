from typing import Literal

from pydantic import BaseModel, model_validator

Provider = Literal["laniqo", "vicomtech", "hkma"]
LanguagePair = Literal["enpl", "eseu", "zhen"]


class TermEntry(BaseModel):
    source: str
    targets: list[str]


class Glossary(BaseModel):
    proper: list[TermEntry]
    random: list[TermEntry]


class TermAnnotation(BaseModel):
    """
    `target` the single lemma annotated for this occurrence
    `target_inflected` the single inflected version that is annotated
    `glossary_targets` all possible alternatives according to the glossary
    """

    source: str
    target: str
    glossary_targets: list[str] = []
    target_inflected: str | None = None


class BitextSample(BaseModel):
    """The Task 2 provided samples"""

    source: str
    target: str
    document_id: str | None = None
    terms: list[TermAnnotation] = []


class Segment(BaseModel):
    """`seen_as_sample`: the segment's gold pair was released as a track-2 sample bitext."""

    source: str
    reference: str | None = None
    seen_as_sample: bool = False
    terms: list[TermAnnotation] = []


class Paragraph(BaseModel):
    """`reference` is None where no gold translation of the released text exists; such
    paragraphs are excluded from scoring on both sides."""

    source: str
    reference: str | None = None
    segments: list[Segment] = []


class Document(BaseModel):
    document_id: str
    paragraphs: list[Paragraph]

    def source_text(self, delimiter: str) -> str:
        return delimiter.join(p.source for p in self.paragraphs)


class TestSet(BaseModel):
    """`glossary` is the released dictionary for track 1; for track 2 it is the evaluation-side
    dictionary the term annotations draw on (hkma), never seen by competitors."""

    provider: Provider
    track: Literal[1, 2]
    pair: LanguagePair
    domain: str
    source_lang: str
    target_lang: str
    paragraph_delimiter: str | None
    documents: list[Document]
    glossary: Glossary | None = None
    samples: list[BitextSample] | None = None

    @model_validator(mode="after")
    def _track_payload(self) -> "TestSet":
        if self.track == 1 and self.glossary is None:
            raise ValueError("track 1 requires a glossary")
        if self.track != 1 and self.samples is None:
            raise ValueError("track 2 requires samples")
        return self

    @model_validator(mode="after")
    def _segments_join_to_paragraphs(self) -> "TestSet":
        """Segments are exact slices of their paragraph on both sides, so anything scored at
        segment level scores the released wording. Chinese joins without a separator."""
        joiners = {"source": "" if self.source_lang == "zh" else " ", "reference": "" if self.target_lang == "zh" else " "}
        for document in self.documents:
            for index, paragraph in enumerate(document.paragraphs):
                for side, joiner in joiners.items():
                    text = getattr(paragraph, side)
                    if text is None or not paragraph.segments:
                        continue
                    joined = joiner.join(getattr(segment, side) or "" for segment in paragraph.segments)
                    if joined != text.strip():
                        raise ValueError(f"{document.document_id} paragraph {index}: segments do not join to the {side}")
        return self

    def public_texts(self) -> list[str]:
        """The released `text.{domain}.{pair}.json` content."""
        if self.paragraph_delimiter is None:
            assert all(len(d.paragraphs) == 1 for d in self.documents), "delimiter-less set must be single-paragraph"
            return [d.paragraphs[0].source for d in self.documents]
        return [d.source_text(self.paragraph_delimiter) for d in self.documents]

    def public_terms(self) -> dict[str, dict[str, list[str]]]:
        """The released `terms.{domain}.{pair}.json` content."""
        assert self.glossary is not None
        return {
            mode: {e.source: e.targets for e in entries}
            for mode, entries in (("proper", self.glossary.proper), ("random", self.glossary.random))
        }

    def public_samples(self) -> list[dict[str, str]]:
        """The released `sample.{domain}.{pair}.json` content."""
        assert self.samples is not None
        return [{self.source_lang: s.source, self.target_lang: s.target} for s in self.samples]
