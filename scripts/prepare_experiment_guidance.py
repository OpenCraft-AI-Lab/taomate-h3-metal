# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Prepare unchanged official Base10 artifacts for guarded Metal A/B experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from taomate_h3.config import DirectRunConfig
from taomate_h3.teacher import launch_audio_guidance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--guidance', type=Path, required=True)
    args = parser.parse_args()
    config = DirectRunConfig.from_dict(json.loads(args.config.read_text()))
    if config.backend != 'mps':
        parser.error('this experiment harness requires MPS')
    launch_audio_guidance(config, output_dir=args.guidance)
    completion = args.guidance / 'complete.json'
    if not completion.is_file():
        raise RuntimeError('teacher did not publish its completion record')
    print(json.dumps({'teacher_artifacts_complete': True,
                      'geometry': [config.width, config.height],
                      'request_count': len(config.prompts),
                      'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(args.guidance.glob('*'))
                                          if p.is_file()}}, indent=2))


if __name__ == '__main__':
    main()
