import json
from pathlib import Path

from pydantic import BaseModel

from wmt26_terminology.schema import TestSet

TRACK_MODES = {1: ("noterm", "proper", "random"), 2: ("noterm", "sample")}


class Submission(BaseModel):
    system: str
    mode: str
    documents: list[list[str]]


def split_paragraphs(text: str, delimiter: str | None) -> list[str]:
    """Split by the test set's own delimiter; document-level sets (no delimiter) are one paragraph.

    Stricter than the official validation-26.py heuristic (\\n\\n when present, else \\n), whose
    per-document delimiter flip lets empty paragraphs and mixed joins mis-split a document while
    still passing the count check."""
    return text.split(delimiter) if delimiter else [text]


def parse_filename(name: str) -> tuple[str, str, str, str]:
    """`{system}.{mode}.{domain}.{pair}.json`; the system name may itself contain dots."""
    stem = name.removesuffix(".json")
    system, mode, domain, pair = stem.rsplit(".", 3)
    return system, mode, domain, pair


def load_submission(path: Path, test_set: TestSet) -> Submission:
    system, mode, _, _ = parse_filename(path.name)
    return parse_submission(json.loads(path.read_text(encoding="utf-8")), test_set, system, mode, path.name)


def parse_submission(raw: list[str], test_set: TestSet, system: str, mode: str, label: str = "submission") -> Submission:
    if mode not in TRACK_MODES[test_set.track]:
        raise ValueError(f"{label}: mode {mode!r} is not a track {test_set.track} mode")
    if len(raw) != len(test_set.documents):
        raise ValueError(f"{label}: {len(raw)} documents, expected {len(test_set.documents)}")
    delimiter = test_set.paragraph_delimiter
    documents = []
    for index, (text, doc) in enumerate(zip(raw, test_set.documents, strict=True)):
        paragraphs = split_paragraphs(text, delimiter)
        if len(paragraphs) != len(doc.paragraphs):
            message = f"{label} doc {index}: {len(paragraphs)} paragraphs, expected {len(doc.paragraphs)}; " + (
                f"paragraphs must be joined with {delimiter!r}" if delimiter else "submit each document as one string"
            )
            if delimiter:
                other = "\n" if delimiter == "\n\n" else "\n\n"
                if len(text.split(other)) == len(doc.paragraphs):
                    message += f" (this document looks {other!r}-joined)"
            raise ValueError(message)
        documents.append(paragraphs)
    return Submission(system=system, mode=mode, documents=documents)


def gold_submission(test_set: TestSet) -> Submission:
    mode = "proper" if test_set.track == 1 else "sample"
    return Submission(
        system="gold", mode=mode, documents=[[p.reference or "" for p in d.paragraphs] for d in test_set.documents]
    )
