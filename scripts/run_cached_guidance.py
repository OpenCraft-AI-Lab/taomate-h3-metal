# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Run the official runner with a previously verified, unchanged teacher artifact.

Experimental A/B harness only: receipts identify artifact reuse, so this wall
clock is Stage3 + publication and does not include teacher production time.
"""
import argparse
import hashlib
import json
from pathlib import Path

from taomate_h3.config import DirectRunConfig
from taomate_h3.runner import run_direct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    args = parser.parse_args()
    config = DirectRunConfig.from_dict(json.loads(args.config.read_text()))
    if config.backend != "mps":
        parser.error("this local experiment harness is for MPS")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n"
    )
    receipt = {"teacher_reused": True, "guidance_root": str(args.guidance.resolve()),
               "guidance_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(args.guidance.glob("*")) if p.is_file()}}
    (config.output_dir / "teacher-reuse.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run_direct(config, audio_guidance_dir=args.guidance)


if __name__ == "__main__":
    main()
