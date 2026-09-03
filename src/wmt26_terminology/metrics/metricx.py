# /// script
# requires-python = ">=3.12"
# dependencies = ["wmt26-terminology-evaluation[neural]"]
# ///
"""MetricX-24 hybrid scorer on aligned segment units, reference-based (`metricx`) or
reference-free (`metricx_qe`). Lower is better, range 0 to 25.

    WMT26_API=... WMT26_SCORER_KEY=... python -m wmt26_terminology.metrics.metricx --metric metricx [--dry-run]
"""

import argparse
import copy
import platform

import torch
from torch import nn
from transformers import T5Tokenizer
from transformers.models.mt5.modeling_mt5 import MT5Config, MT5PreTrainedModel, MT5Stack

from wmt26_terminology.metrics.scorer_api import ScoreFn, ScorerClient
from wmt26_terminology.models import METRICX_24_XL, MT5_TOKENIZER, fetch, fetch_snapshot

NAME = "metricx"
EXTERNAL = True
WORST_SCORE = 25.0
MAX_INPUT_LENGTH = 1536
VERSIONS = {"metricx": "metricx-24-xl-seg", "metricx_qe": "metricx-24-xl-qe-seg"}
# google-research/metricx, commit fc4978eb064670f7cc33e93ea4f52d38396b8ae6 (Apache 2.0)
_UPSTREAM = "https://github.com/google-research/metricx/tree/fc4978eb064670f7cc33e93ea4f52d38396b8ae6"
_SCORE_TOKEN = 250089
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class MT5ForRegression(MT5PreTrainedModel):
    """MetricX's regression head, ported from metricx24/models.py at the pinned upstream
    commit: the encoder output is decoded for a single step and the logit of one fixed
    vocabulary entry is the score."""

    def __init__(self, config: MT5Config) -> None:
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        decoder_input_ids = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=hidden_states.device)
        sequence_output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output *= self.model_dim**-0.5
        return torch.clamp(self.lm_head(sequence_output)[:, 0, _SCORE_TOKEN], 0, WORST_SCORE)


def input_text(unit: dict, qe: bool) -> str:
    text = f"source: {unit['source']} candidate: {unit['hypothesis']}"
    return text if qe else f"{text} reference: {unit['reference']}"


class MetricX:
    def __init__(self, precision: str) -> None:
        self.tokenizer = T5Tokenizer(vocab_file=str(fetch(MT5_TOKENIZER)))
        self.model = MT5ForRegression.from_pretrained(fetch_snapshot(METRICX_24_XL), torch_dtype=_DTYPES[precision])
        self.model.eval().to("cuda")

    def encode(self, text: str) -> list[int]:
        # Upstream tokenizes with truncation at MAX_INPUT_LENGTH and drops the trailing EOS.
        return self.tokenizer(text, max_length=MAX_INPUT_LENGTH, truncation=True)["input_ids"][:-1]

    @torch.inference_mode()
    def score(self, texts: list[str], batch_size: int) -> list[float]:
        encoded = [self.encode(t) for t in texts]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        values = [0.0] * len(encoded)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            width = max(len(encoded[i]) for i in batch)
            ids = torch.full((len(batch), width), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                ids[row, : len(encoded[i])] = torch.tensor(encoded[i])
                mask[row, : len(encoded[i])] = 1
            predictions = self.model(ids.to("cuda"), mask.to("cuda"))
            for row, i in enumerate(batch):
                values[i] = float(predictions[row])
        return values


def scorer(model: MetricX, qe: bool, batch_size: int) -> ScoreFn:
    def score(units: list[dict]) -> list[dict]:
        values = model.score([input_text(u, qe) for u in units], batch_size) if units else []
        return [{"id": u["id"], "value": v} for u, v in zip(units, values, strict=True)]

    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=sorted(VERSIONS), default="metricx")
    parser.add_argument("--precision", choices=sorted(_DTYPES), default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fetch", type=int, default=2000, help="units per API round trip")
    parser.add_argument("--direction", default="", help="score one language direction only, e.g. eseu")
    parser.add_argument(
        "--order", choices=("oldest", "newest"), default="oldest", help="which end of the file list to start from"
    )
    parser.add_argument("--dry-run", action="store_true", help="time one file, post nothing")
    args = parser.parse_args()
    qe = args.metric == "metricx_qe"
    score = scorer(MetricX(args.precision), qe, args.batch_size)
    client = ScorerClient(args.metric, VERSIONS[args.metric], args.direction, args.order)
    if args.dry_run:
        client.dry_run(score, limit=args.fetch)
        return
    meta = {
        "model": METRICX_24_XL[0].repo,
        "revision": METRICX_24_XL[0].revision,
        "tokenizer": f"{MT5_TOKENIZER.repo}@{MT5_TOKENIZER.revision}",
        "upstream": _UPSTREAM,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "precision": args.precision,
        "max_input_length": MAX_INPUT_LENGTH,
        "qe": qe,
        "level": "segment",
        "forced": f"empty and over-cap pieces take {WORST_SCORE}",
    }
    total = client.run(score, meta, WORST_SCORE, limit=args.fetch)
    print(f"done, {total} unit scores posted")


if __name__ == "__main__":
    main()
