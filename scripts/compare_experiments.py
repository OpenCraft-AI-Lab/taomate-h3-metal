# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Compare delivered frames/audio and optional decoder-input latents without GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def probe(path):
    return json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        text=True,
    ))


def decoder(path, mapping, output_format):
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", mapping, *output_format, "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def read_exact(stream, size):
    chunks, count = [], 0
    while count < size:
        part = stream.read(size - count)
        if not part:
            break
        chunks.append(part)
        count += len(part)
    return b"".join(chunks)


def compare_video(a, b, frame_bytes):
    processes = [decoder(p, "0:v:0", ["-pix_fmt", "rgb24", "-f", "rawvideo"]) for p in (a, b)]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    frames = identical = elements = total_absolute = total_squared = maximum = 0
    try:
        while True:
            data = [read_exact(p.stdout, frame_bytes) for p in processes]
            if not any(data):
                break
            if any(len(part) != frame_bytes for part in data):
                raise RuntimeError("video lengths differ or partial decoded frame")
            for digest, part in zip(hashes, data):
                digest.update(part)
            frames += 1
            identical += data[0] == data[1]
            delta = np.frombuffer(data[0], np.uint8).astype(np.int16)
            delta -= np.frombuffer(data[1], np.uint8).astype(np.int16)
            total_absolute += int(np.abs(delta).sum(dtype=np.int64))
            total_squared += int((delta.astype(np.int32)**2).sum(dtype=np.int64))
            maximum = max(maximum, int(np.abs(delta).max()))
            elements += delta.size
        for process in processes:
            errors = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(errors)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    mse = total_squared / elements
    return {"frames": frames, "identical_frames": identical,
            "decoded_rgb_sha256": [h.hexdigest() for h in hashes],
            "mean_absolute_pixel_error": total_absolute/elements, "max_pixel_error": maximum,
            "psnr_db": 10*math.log10(255**2/mse) if mse else None,
            "exact_pixels": total_squared == 0}


def compare_audio(a, b):
    arrays, hashes = [], []
    for path in (a, b):
        process = decoder(path, "0:a:0", ["-acodec", "pcm_f32le", "-f", "f32le"])
        data, errors = process.communicate()
        if process.returncode:
            raise RuntimeError(errors.decode())
        arrays.append(np.frombuffer(data, np.float32))
        hashes.append(hashlib.sha256(data).hexdigest())
    if arrays[0].shape != arrays[1].shape:
        raise RuntimeError("audio sample lengths differ")
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("decoded audio contains nonfinite samples")
    delta = arrays[1].astype(np.float64)-arrays[0]
    return {"decoded_f32_sha256": hashes, "samples_including_channels": arrays[0].size,
            "exact_samples": np.array_equal(arrays[0], arrays[1]),
            "rms_error": float(np.sqrt(np.mean(delta**2))),
            "maximum_absolute_error": float(np.max(np.abs(delta))),
            "rms": [float(np.sqrt(np.mean(array.astype(np.float64)**2))) for array in arrays]}


def compare_latents(a, b):
    from safetensors.numpy import load_file

    tensors = [load_file(str(p)) for p in (a, b)]
    results = {}
    for key in tensors[0]:
        aa, bb = [values[key].astype(np.float64) for values in tensors]
        if aa.shape != bb.shape or not (np.isfinite(aa).all() and np.isfinite(bb).all()):
            raise RuntimeError("latent shape mismatch or nonfinite values")
        difference = bb-aa
        results[key] = {"shape": list(aa.shape), "exact": np.array_equal(aa, bb),
                        "relative_l2": float(np.linalg.norm(difference)/np.linalg.norm(aa)),
                        "rms_error": float(np.sqrt(np.mean(difference**2))),
                        "max_absolute_error": float(np.max(np.abs(difference)))}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = [probe(p) for p in (args.baseline, args.candidate)]
    streams = [next(s for s in m["streams"] if s["codec_type"] == "video") for m in metadata]
    if any(streams[0][key] != streams[1][key] for key in ("width", "height", "r_frame_rate")):
        raise RuntimeError("video geometries differ")
    result = {"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve()),
              "file_sha256": [hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.baseline, args.candidate)],
              "video": compare_video(args.baseline, args.candidate,
                                     streams[0]["width"]*streams[0]["height"]*3),
              "audio": compare_audio(args.baseline, args.candidate),
              "note": "Pixel/latent distance is not a perceptual quality or semantic parity score."}
    latents = [p.parent / "publication-latents.safetensors" for p in (
        args.baseline, args.candidate
    )]
    if all(p.exists() for p in latents):
        result["latents"] = compare_latents(*latents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
