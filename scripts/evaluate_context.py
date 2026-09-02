#!/usr/bin/env python3
"""
Context-aware voice evaluation script.

Generates a standard evaluation set for one identity across multiple contexts
and records generation metadata for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate context-aware voice generation")
    parser.add_argument("--identity-id", required=True, help="Voice identity ID")
    parser.add_argument("--expression", default="neutral", help="Base expression preset")
    parser.add_argument("--output-dir", type=Path, help="Directory for evaluation outputs")
    parser.add_argument("--home", type=Path, help="Override VOICECLONE_HOME")
    args = parser.parse_args()

    if args.home:
        os.environ["VOICECLONE_HOME"] = str(args.home)
        import voiceclone.Config as config
        config.ROOT = args.home
        config.VOICES_DIR = args.home / "voices"
        config.OUTPUTS_DIR = args.home / "outputs"

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.evaluation.render_metadata import load_render_metadata

    service = VoiceIdentityService(auto_migrate=False)
    identity = service.get_identity(args.identity_id)

    contexts = [
        "default",
        "desktop",
        "quiet_environment",
        "noisy_environment",
        "car",
        "presentation",
        "notification",
        "warning",
    ]
    sentences = [
        "We need to leave now.",
        "The project is finally complete.",
        "Please pay attention to the road ahead.",
    ]

    out_dir = args.output_dir or (
        Path(os.environ.get("VOICECLONE_HOME", Path.home() / ".voiceclone"))
        / "outputs"
        / identity.id
        / "context_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for context in contexts:
        for sentence in sentences:
            t0 = time.time()
            try:
                output = service.synthesize(
                    identity.id,
                    sentence,
                    expression=args.expression,
                    context=context,
                )
                gen_time = round(time.time() - t0, 2)
                meta = load_render_metadata(output)
                similarity = service.compare(identity.id, output)
                duration = meta.get("duration_seconds", "")
            except Exception as e:
                print(f"FAILED {context}/{sentence[:30]}: {e}", file=sys.stderr)
                continue

            rows.append({
                "identity_id": identity.id,
                "identity_name": identity.name,
                "sentence": sentence,
                "base_expression": meta.get("base_expression_name", ""),
                "context": meta.get("context_name", context),
                "resolved_expression": meta.get("resolved_expression_name", ""),
                "applied_rules": json.dumps(meta.get("applied_context_rules", [])),
                "context_policy_version": meta.get("context_policy_version", ""),
                "renderer_version": meta.get("renderer_version", ""),
                "generation_parameters": json.dumps(meta.get("generation_parameters", {})),
                "speaker_similarity": round(similarity, 4),
                "duration_seconds": duration,
                "gen_time_s": gen_time,
                "output": output,
                "identity_fidelity": "",
                "intelligibility": "",
                "expressive_distinction": "",
                "naturalness": "",
                "artifact_presence": "",
            })

    if not rows:
        print("No outputs generated.", file=sys.stderr)
        return 1

    csv_path = out_dir / f"context_eval_{int(time.time())}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
