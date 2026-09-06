#!/usr/bin/env python3
"""Basic Phase 4 memory evaluation report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from voiceclone.core.service import VoiceIdentityService
from voiceclone.evaluation.render_metadata import load_render_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-id", required=True)
    parser.add_argument("--text", default="Minitorch accelerates research.")
    args = parser.parse_args()

    service = VoiceIdentityService()
    identity = service.get_identity(args.identity_id)

    rows = []
    for label, use_memory in (("baseline", False), ("memory_enabled", True)):
        t0 = time.time()
        path = service.synthesize(identity.id, args.text, use_memory=use_memory)
        elapsed = round(time.time() - t0, 3)
        meta = load_render_metadata(path)
        sim = service.compare(identity.id, path)
        rows.append(
            {
                "mode": label,
                "identity": identity.id,
                "memories_used": meta.get("memory_items_used", []),
                "expression": meta.get("base_expression_name"),
                "context": meta.get("context_name"),
                "resolved_expression": meta.get("resolved_expression_name"),
                "render_text": meta.get("render_text"),
                "generation_time_s": elapsed,
                "similarity": round(sim, 4),
            }
        )

    print(json.dumps({"identity": identity.id, "comparisons": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
