from pydantic import BaseModel

Span = tuple[int, int]


class AnnotationResult(BaseModel):
    """Where one required term occurrence was found in the hypothesis paragraph. `spans`
    holds the first match per tier (`annotated`, `inflected`, `annotated_lemma`, `glossary`,
    `glossary_lemma`), `exclusive` the span assigned by the disjoint matching over all tiers."""

    source: str
    target: str
    seen_as_sample: bool = False
    in_samples: bool = False
    has_inflected: bool = False
    spans: dict[str, Span] = {}
    exclusive: Span | None = None


class TermRates(BaseModel):
    """Rates over required occurrences: `annotated` accepts the annotated lemma or the
    inflected form, `glossary` additionally any glossary alternative, `*_lemma` the same
    in lemma space, `exclusive` requires pairwise disjoint spans over all of it.
    `*_types` count a (document, source term) once; `inflected_exact` is conditional on
    occurrences that carry an inflected form."""

    occurrences: int
    types: int
    occurrences_with_inflected: int
    annotated: float
    annotated_lemma: float
    glossary: float
    glossary_lemma: float
    exclusive: float
    inflected_exact: float
    annotated_lemma_types: float
    glossary_lemma_types: float


class ParagraphResult(BaseModel):
    document_index: int
    paragraph_index: int
    scored: bool
    chrf: float | None = None
    annotations: list[AnnotationResult] = []


class EvaluationResult(BaseModel):
    system: str
    mode: str
    provider: str
    domain: str
    pair: str
    track: int
    metric_version: str
    document_chrf: float
    paragraph_chrf: float
    terms: dict[str, TermRates]
    paragraphs: list[ParagraphResult]
