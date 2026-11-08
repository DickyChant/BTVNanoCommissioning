"""Check the narration by ear, without ears: transcribe the narrated video with faster-whisper and compare the words with the
written narration (voice/narration.json); prints the match ratio and every stretch that differs (a mispronounced term shows up
as a substitution). usage (the environment of voice/pixi.toml):  python check_voice.py <narrated.mp4> <voice dir> [--model small.en]"""
import argparse
import difflib
import json
import os
import re

from faster_whisper import WhisperModel


def words(s):
    s = s.lower().replace("j/ψ", " jpsi ").replace("j/psi", " jpsi ").replace("j psi", " jpsi ").replace("ψ", "psi")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 12,000 -> 12000
    s = re.sub(r"[^a-z0-9.%]+", " ", s.replace("-", " "))
    return [w.strip(".") for w in s.split() if w.strip(".")]


ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("voice")
ap.add_argument("--model", default="small.en")
args = ap.parse_args()
narr = json.load(open(os.path.join(args.voice, "narration.json")))
model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
segs, info = model.transcribe(args.video, language="en", beam_size=5, vad_filter=True)
heard = " ".join(s.text for s in segs)
open(os.path.join(args.voice, "transcript.txt"), "w").write(heard + "\n")
ref, hyp = words(" ".join(narr.values())), words(heard)
sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
print(f"{len(ref)} words written, {len(hyp)} heard; match ratio {sm.ratio():.3f}")
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op != "equal":
        print(f"  {op:8} written: {' '.join(ref[max(i1 - 2, 0):i2 + 2])!r:60}  heard: {' '.join(hyp[max(j1 - 2, 0):j2 + 2])!r}")
