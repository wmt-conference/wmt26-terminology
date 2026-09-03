import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

_VERIFIED = Path.home() / ".cache" / "wmt26_terminology" / "verified"


@dataclass(frozen=True)
class Artifact:
    """One file of a Hugging Face repository at a fixed revision. Scoring reads exactly these
    bytes, so a rerun anywhere reproduces the published numbers."""

    repo: str
    revision: str
    filename: str
    sha256: str


XLMR_TOKENIZER = Artifact(
    "xlm-roberta-large",
    "c23d21b0620b635a76227c604d44e43a9f0ee389",
    "sentencepiece.bpe.model",
    "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
)
MT5_TOKENIZER = Artifact(
    "google/mt5-xl",
    "63fc6450d80515b48e026b69ef2fbbd426433e84",
    "spiece.model",
    "ef78f86560d809067d12bac6c09f19a462cb3af3f54d2b8acbba26e1433125d6",
)
COMET_DA = (
    Artifact(
        "Unbabel/wmt22-comet-da",
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2",
        "checkpoints/model.ckpt",
        "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af",
    ),
    Artifact(
        "Unbabel/wmt22-comet-da",
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2",
        "hparams.yaml",
        "265ef22345ea5b9ffa020a7fe5be613a95ff931c44bf8d09a26d96c6c6048f60",
    ),
)
COMET_KIWI = (
    Artifact(
        "Unbabel/wmt22-cometkiwi-da",
        "1ad785194e391eebc6c53e2d0776cada8f83179a",
        "checkpoints/model.ckpt",
        "4f357aa38b0737dcd502f166238c99711ff3419d7b5c8cdf9cde08525a8e7858",
    ),
    Artifact(
        "Unbabel/wmt22-cometkiwi-da",
        "1ad785194e391eebc6c53e2d0776cada8f83179a",
        "hparams.yaml",
        "eee0f391b4e2baee489117e59d967a9be0b1ad027556152c6cefcd41039c778c",
    ),
)
XCOMET_XXL = (
    Artifact(
        "Unbabel/XCOMET-XXL",
        "873bac1b1c461e410c4a6e379f6790d3d1c7c214",
        "checkpoints/model.ckpt",
        "e760e1f568af69b7a1bf7aeb46d8f3be21f01be7cbda480f8225ee81eb0af27a",
    ),
    Artifact(
        "Unbabel/XCOMET-XXL",
        "873bac1b1c461e410c4a6e379f6790d3d1c7c214",
        "hparams.yaml",
        "0519fd6b5ad74bb15c87894b2b862e1a005219939ad2e474e63eeff5aa6b2214",
    ),
)
METRICX_24_XL = tuple(
    Artifact("google/metricx-24-hybrid-xl-v2p6", "f6e7f99a655582f28cb998dd3e6ca86b4217430d", filename, sha256)
    for filename, sha256 in (
        ("config.json", "47b47c4c4a892f2c7411a0e7711ad76f5bdcd13c176161e04f26b0dc681b4c42"),
        ("pytorch_model.bin.index.json", "e00c00362f242bd766e84a3a30d820e80ba3be1c08eab2b9bf123535d5fe1c92"),
        ("pytorch_model-00001-of-00002.bin", "f0481c16fa7171bbb1a724cce3b8598b8d78bb8e8209138fecf7fdedd5200327"),
        ("pytorch_model-00002-of-00002.bin", "24978065a17c3d63617d92f093a1b4cdf3b94cb68b969ebd3c947c334d837cbc"),
    )
)


def fetch(artifact: Artifact) -> Path:
    """Download (or reuse from the Hugging Face cache) and verify once per checksum; the
    verification marker spares re-hashing multi-gigabyte checkpoints on every run."""
    path = Path(hf_hub_download(artifact.repo, artifact.filename, revision=artifact.revision))
    marker = _VERIFIED / artifact.sha256
    if not marker.exists():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise RuntimeError(f"{artifact.repo}/{artifact.filename}: sha256 {digest.hexdigest()} != {artifact.sha256}")
        _VERIFIED.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(path), encoding="utf-8")
    return path


def fetch_snapshot(artifacts: tuple[Artifact, ...]) -> Path:
    """The snapshot directory holding every artifact of one model, as loaders expect it."""
    paths = [fetch(a) for a in artifacts]
    depth = artifacts[0].filename.count("/")
    return paths[0].parents[depth]
