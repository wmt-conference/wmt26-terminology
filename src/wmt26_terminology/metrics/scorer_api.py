import os
import time
from collections import defaultdict
from collections.abc import Callable

import httpx

Scores = list[dict]
ScoreFn = Callable[[list[dict]], Scores]


class ScorerClient:
    """Fetch-score-post loop against the portal's external scorer API. Units whose status is
    not ok never reach the model: they are posted with the metric's worst score and the status
    as payload, so omissions and over-cap pieces count against the system."""

    def __init__(self, metric: str, version: str, direction: str = "", order: str = "oldest") -> None:
        self.metric, self.version, self.direction, self.order = metric, version, direction, order
        api = os.environ.get("WMT26_API", "http://127.0.0.1:8092")
        key = os.environ["WMT26_SCORER_KEY"]
        self._client = httpx.Client(base_url=api, headers={"authorization": f"Bearer {key}"}, timeout=600)

    def units(self, level: str = "segment", limit: int = 1000) -> list[dict]:
        params = {"metric": self.metric, "version": self.version, "level": level, "limit": limit}
        params |= {"direction": self.direction, "order": self.order}
        response = self._client.get("/v1/external/units", params=params)
        response.raise_for_status()
        return response.json()

    def post(self, scores: Scores, meta: dict) -> int:
        payload = {"metric": self.metric, "version": self.version, "meta": meta, "scores": scores}
        response = self._client.post("/v1/external/scores", json=payload)
        response.raise_for_status()
        return response.json()["written"]

    def reset(self, system: str = "") -> dict:
        params = {"metric": self.metric, "version": self.version, "system": system}
        response = self._client.delete("/v1/external/scores", params=params)
        response.raise_for_status()
        return response.json()

    def run(self, score: ScoreFn, meta: dict, worst: float, level: str = "segment", limit: int = 1000) -> int:
        total = 0
        while True:
            units = self.units(level, limit)
            if not units:
                return total
            forced = [{"id": u["id"], "value": worst, "payload": {"forced": u["status"]}} for u in units if u["status"] != "ok"]
            scored = score([u for u in units if u["status"] == "ok"])
            total += self.post(forced + scored, meta)
            first = units[0]
            print(f"posted {total} so far ({first['system']} {first['mode']}.{first['domain']}.{first['direction']})")

    def dry_run(self, score: ScoreFn, level: str = "segment", limit: int = 1000) -> None:
        """Time one file's worth of units end to end; nothing is posted."""
        by_file: dict[str, list[dict]] = defaultdict(list)
        for u in self.units(level, limit):
            by_file[u["id"].split(":")[0]].append(u)
        units = next(iter(by_file.values()))
        ok = [u for u in units if u["status"] == "ok"]
        label = f"{units[0]['system']} {units[0]['mode']}.{units[0]['domain']}.{units[0]['direction']}"
        started = time.time()
        scores = score(ok)
        elapsed = time.time() - started
        per_unit = elapsed / max(len(ok), 1) * 1000
        print(f"{label}: {len(units)} units, {len(ok)} scored in {elapsed:.1f}s ({per_unit:.0f} ms/unit)")
        print("sample values:", [round(s["value"], 4) for s in scores[:8]])
