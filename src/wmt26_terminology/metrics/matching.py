import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmt26_terminology.metrics.lemma import LemmaView

_ZERO_WIDTH = dict.fromkeys(map(ord, "​﻿"), None)
_NODE_BUDGET = 1_000_000


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_ZERO_WIDTH)).strip().lower()


def char_spans(text: str, form: str) -> list[tuple[int, int]]:
    """Word-boundary-anchored occurrences (unicode `\\w`, so accented letters block matches)."""
    if not form:
        return []
    pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def _span_mask(text_length: int, start: int, end: int) -> int:
    return ((1 << (end - start)) - 1) << (text_length - end)


class MaskSpace:
    """Character masks over one hypothesis paragraph, surface and lemma space combined: a
    match in either space also blocks the mapped span in the other, so exclusivity holds
    across spaces. Each mask remembers the surface span it stands for."""

    def __init__(self, surface: str, view: "LemmaView | None") -> None:
        self.surface = surface
        self.view = view
        self._shift = len(surface) + 1
        self.spans: dict[int, tuple[int, int]] = {}

    def surface_masks(self, form: str) -> list[int]:
        masks = []
        for start, end in char_spans(self.surface, form):
            mask = _span_mask(len(self.surface), start, end)
            if self.view is not None:
                for span in {self.view.to_lemma[i] for i in range(start, end)} - {None}:
                    mask |= _span_mask(len(self.view.lemma_text), *span) << self._shift
            self.spans.setdefault(mask, (start, end))
            masks.append(mask)
        return masks

    def lemma_masks(self, lemma_form: str) -> list[int]:
        if self.view is None:
            return []
        masks = []
        for start, end in char_spans(self.view.lemma_text, lemma_form):
            mask = _span_mask(len(self.view.lemma_text), start, end) << self._shift
            surface_spans = sorted({self.view.to_surface[i] for i in range(start, end)} - {None})
            for span in surface_spans:
                mask |= _span_mask(len(self.surface), *span)
            if surface_spans:
                self.spans.setdefault(mask, (surface_spans[0][0], surface_spans[-1][1]))
            masks.append(mask)
        return masks


def _grouped(requirements: list[list[int]]) -> list[tuple[list[int], list[int]]]:
    """(candidate masks, requirement indices) per distinct candidate set, smallest sets first."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, requirement in enumerate(requirements):
        if requirement:
            groups.setdefault(tuple(sorted(set(requirement))), []).append(index)
    return sorted(((list(masks), members) for masks, members in groups.items()), key=lambda item: len(item[0]))


def _greedy(items: list[tuple[list[int], list[int]]]) -> list[list[int]]:
    chosen: list[list[int]] = [[] for _ in items]
    used = 0
    for group, (masks, members) in enumerate(items):
        for mask in masks:
            if len(chosen[group]) == len(members):
                break
            if used & mask == 0:
                used |= mask
                chosen[group].append(mask)
    return chosen


def max_disjoint(requirements: list[list[int]]) -> list[int | None]:
    """An assignment of pairwise disjoint masks satisfying as many requirements as possible
    (match_accuracy's exhaustive search, made tractable: identical requirement sets are
    assigned as combinations, a greedy assignment seeds the bound, a node budget caps
    pathological instances). Returns the chosen mask per requirement, None where unmet."""
    items = _grouped(requirements)
    suffix = [0] * (len(items) + 1)
    for i in range(len(items) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + min(len(items[i][0]), len(items[i][1]))
    total = sum(len(members) for _, members in items)
    best = _greedy(items)
    best_count = sum(len(b) for b in best)
    budget = _NODE_BUDGET
    chosen: list[list[int]] = [[] for _ in items]

    def descend(group: int, mask_index: int, used: int, count: int) -> None:
        nonlocal best, best_count, budget
        if count > best_count:
            best_count, best = count, [list(c) for c in chosen]
        if best_count == total or budget <= 0:
            return
        budget -= 1
        if group == len(items):
            return
        masks, members = items[group]
        picks_left = len(members) - len(chosen[group])
        if picks_left == 0 or mask_index == len(masks):
            descend(group + 1, 0, used, count)
            return
        if count + min(picks_left, len(masks) - mask_index) + suffix[group + 1] <= best_count:
            return
        mask = masks[mask_index]
        if used & mask == 0:
            chosen[group].append(mask)
            descend(group, mask_index + 1, used | mask, count + 1)
            chosen[group].pop()
        descend(group, mask_index + 1, used, count)

    if items:
        descend(0, 0, 0, 0)
    assignment: list[int | None] = [None] * len(requirements)
    for (_, members), masks in zip(items, best, strict=True):
        for member, mask in zip(members, masks, strict=False):
            assignment[member] = mask
    return assignment
