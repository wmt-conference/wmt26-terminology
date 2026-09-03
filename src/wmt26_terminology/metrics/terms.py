from typing import TYPE_CHECKING

from wmt26_terminology.metrics.matching import MaskSpace, max_disjoint, normalize
from wmt26_terminology.results import AnnotationResult, TermRates
from wmt26_terminology.schema import TermAnnotation, TestSet
from wmt26_terminology.submission import Submission

if TYPE_CHECKING:
    from wmt26_terminology.metrics.lemma import Lemmatizer

SURFACE_TIERS = ("annotated", "inflected", "glossary")
LEMMA_TIERS = ("annotated_lemma", "glossary_lemma")
ANNOTATED_TIERS = ("annotated", "inflected", "annotated_lemma")


def _forms(annotation: TermAnnotation) -> dict[str, list[str]]:
    target = normalize(annotation.target)
    inflected = normalize(annotation.target_inflected) if annotation.target_inflected else ""
    glossary = [g for g in (normalize(t) for t in annotation.glossary_targets) if g and g != target]
    return {"annotated": [target], "inflected": [inflected] if inflected else [], "glossary": glossary}


def _lemma_forms(test_set: TestSet, lemmatizer: "Lemmatizer | None") -> dict[str, str]:
    if lemmatizer is None:
        return {}
    forms: set[str] = set()
    for doc in test_set.documents:
        for paragraph in doc.paragraphs:
            for segment in paragraph.segments:
                for annotation in segment.terms:
                    forms.update(f for tier in _forms(annotation).values() for f in tier)
    return lemmatizer.phrases(sorted(forms))


def _sample_terms(test_set: TestSet) -> set[str]:
    return {t.source for sample in test_set.samples or [] for t in sample.terms}


def _score_paragraph(
    space: MaskSpace, annotations: list[tuple[TermAnnotation, bool]], lemma_forms: dict[str, str], sample_terms: set[str]
) -> list[AnnotationResult]:
    results, requirements = [], []
    for annotation, seen in annotations:
        forms = _forms(annotation)
        masks: dict[str, list[int]] = {tier: [m for f in forms[tier] for m in space.surface_masks(f)] for tier in SURFACE_TIERS}
        masks["annotated_lemma"] = [
            m
            for f in forms["annotated"] + forms["inflected"]
            if (lemma := lemma_forms.get(f))
            for m in space.lemma_masks(lemma)
        ]
        masks["glossary_lemma"] = [
            m for f in forms["glossary"] if (lemma := lemma_forms.get(f)) for m in space.lemma_masks(lemma)
        ]
        spans = {tier: space.spans[tier_masks[0]] for tier, tier_masks in masks.items() if tier_masks}
        results.append(
            AnnotationResult(
                source=annotation.source,
                target=annotation.target,
                seen_as_sample=seen,
                in_samples=annotation.source in sample_terms,
                has_inflected=annotation.target_inflected is not None,
                spans=spans,
            )
        )
        requirements.append([m for tier_masks in masks.values() for m in tier_masks])
    for result, mask in zip(results, max_disjoint(requirements), strict=True):
        if mask is not None:
            result.exclusive = space.spans[mask]
    return results


def score_terms(
    test_set: TestSet, submission: Submission, lemmatizer: "Lemmatizer | None" = None
) -> list[list[list[AnnotationResult]]]:
    """Per document, per paragraph: one result per required term occurrence; paragraphs
    without a reference get an empty list."""
    lemma_forms = _lemma_forms(test_set, lemmatizer)
    sample_terms = _sample_terms(test_set)
    hypotheses = [hyp for hyp_doc in submission.documents for hyp in hyp_doc]
    views = lemmatizer.views(hypotheses) if lemmatizer else [None] * len(hypotheses)
    results, index = [], 0
    for doc, hyp_doc in zip(test_set.documents, submission.documents, strict=True):
        doc_results = []
        for paragraph, hyp in zip(doc.paragraphs, hyp_doc, strict=True):
            view = views[index]
            index += 1
            if paragraph.reference is None:
                doc_results.append([])
                continue
            annotations = [(t, segment.seen_as_sample) for segment in paragraph.segments for t in segment.terms]
            doc_results.append(_score_paragraph(MaskSpace(hyp, view), annotations, lemma_forms, sample_terms))
        results.append(doc_results)
    return results


def _rates(items: list[tuple[int, AnnotationResult]]) -> TermRates | None:
    if not items:
        return None
    n = len(items)
    types: dict[tuple[int, str], dict[str, bool]] = {}
    counts = dict.fromkeys(("annotated", "annotated_lemma", "glossary", "glossary_lemma", "exclusive", "inflected"), 0)
    with_inflected = sum(r.has_inflected for _, r in items)
    for doc_index, r in items:
        annotated = any(t in r.spans for t in ("annotated", "inflected"))
        annotated_lemma = annotated or "annotated_lemma" in r.spans
        glossary = annotated or "glossary" in r.spans
        glossary_lemma = annotated_lemma or glossary or "glossary_lemma" in r.spans
        counts["annotated"] += annotated
        counts["annotated_lemma"] += annotated_lemma
        counts["glossary"] += glossary
        counts["glossary_lemma"] += glossary_lemma
        counts["exclusive"] += r.exclusive is not None
        key = (doc_index, r.source)
        entry = types.setdefault(key, {"annotated_lemma": False, "glossary_lemma": False})
        entry["annotated_lemma"] |= annotated_lemma
        entry["glossary_lemma"] |= glossary_lemma
        counts["inflected"] += "inflected" in r.spans
    return TermRates(
        occurrences=n,
        types=len(types),
        occurrences_with_inflected=with_inflected,
        annotated=counts["annotated"] / n,
        annotated_lemma=counts["annotated_lemma"] / n,
        glossary=counts["glossary"] / n,
        glossary_lemma=counts["glossary_lemma"] / n,
        exclusive=counts["exclusive"] / n,
        inflected_exact=counts["inflected"] / with_inflected if with_inflected else 0.0,
        annotated_lemma_types=sum(t["annotated_lemma"] for t in types.values()) / len(types),
        glossary_lemma_types=sum(t["glossary_lemma"] for t in types.values()) / len(types),
    )


def term_rates(results: list[list[list[AnnotationResult]]], track: int) -> dict[str, TermRates]:
    items = [(d, r) for d, doc in enumerate(results) for para in doc for r in para]
    splits = {"all": items}
    if track == 2:  # ruff: ignore[magic-value-comparison]
        splits["seen_as_sample"] = [(d, r) for d, r in items if r.seen_as_sample]
        splits["unseen"] = [(d, r) for d, r in items if not r.seen_as_sample]
        splits["term_in_samples"] = [(d, r) for d, r in items if r.in_samples]
        splits["term_not_in_samples"] = [(d, r) for d, r in items if not r.in_samples]
    return {name: rates for name, subset in splits.items() if (rates := _rates(subset)) is not None}
