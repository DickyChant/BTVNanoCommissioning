"""Narration of the animation: speech for every segment of voice/narration.json (python ttjpsi.py --narration ...), with
Kokoro (local, ONNX; no text leaves the machine). Writes voice/<key>.wav (24 kHz, leading and trailing silence trimmed) and
voice/durations.json, which the scene reads to hold each segment until it has been spoken; a segment is synthesised again
only when its text, the voice or the speed changed.
usage (the environment of voice/pixi.toml, see README):  python voice.py <voice dir> --models <dir of kokoro-v1.0.onnx and
voices-v1.0.bin> [--voice af_heart] [--speed 1.0]"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

# the written form (also the subtitles) -> words the phonemiser says right
# (each checked by synthesising it and transcribing it back with faster-whisper: check_voice.py)
SPEECH = [("J/ψ", "J psi"), ("TeV", "T e V"), ("LETO", "Lee toe"), ("POWHEG", "Powheg"), ("dimuon", "die muon"), ("dilepton", "die lep-tawn"),
          ("lepton", "lep-tawn"), ("parton", "par-tawn")]
YEARS = {"2022": "twenty twenty-two", "2023": "twenty twenty-three", "2024": "twenty twenty-four", "2025": "twenty twenty-five",
         "2026": "twenty twenty-six"}


def spoken(text):
    for a, b in SPEECH:
        text = text.replace(a, b)
    text = re.sub(r"\bb-(?=[a-z])", "b ", text)  # "b-decay" is heard as "B2K"
    return re.sub(r"\b(202[2-6])\b", lambda m: YEARS[m.group(1)], text)


def trim(x, sr, thr=0.01, pad=0.05):
    """cut the silence at both ends, keeping `pad` s"""
    loud = np.flatnonzero(np.abs(x) > thr)
    if not len(loud):
        return x
    p = int(pad * sr)
    return x[max(loud[0] - p, 0):loud[-1] + p]


ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--models", required=True)
ap.add_argument("--voice", default="af_heart")
ap.add_argument("--speed", type=float, default=1.0)
args = ap.parse_args()
narr = json.load(open(os.path.join(args.dir, "narration.json")))
dp = os.path.join(args.dir, "durations.json")
old = json.load(open(dp)) if os.path.exists(dp) else {}
tts = None
out = {}
for key, text in narr.items():
    say = spoken(text)
    sha = hashlib.sha1(f"{say}|{args.voice}|{args.speed}".encode()).hexdigest()[:16]
    wav = os.path.join(args.dir, f"{key}.wav")
    if old.get(key, {}).get("sha") == sha and os.path.exists(wav):
        out[key] = old[key]
        continue
    if tts is None:
        tts = Kokoro(os.path.join(args.models, "kokoro-v1.0.onnx"), os.path.join(args.models, "voices-v1.0.bin"))
    x, sr = tts.create(say, voice=args.voice, speed=args.speed, lang="en-us")
    x = trim(np.asarray(x, dtype=np.float32), sr)
    sf.write(wav, x, sr, subtype="PCM_16")
    out[key] = {"dur": round(len(x) / sr, 3), "sha": sha, "sr": sr}
    print(f"{key:8} {len(x) / sr:5.1f} s  {say}", flush=True)
json.dump(out, open(dp, "w"), indent=1)
print(f"{len(out)} segments, {sum(v['dur'] for v in out.values()):.0f} s of speech -> {dp}")
