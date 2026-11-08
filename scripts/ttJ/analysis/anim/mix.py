"""Lay the narration on the joined video: every segment voice/<key>.wav starts VOICE_LEAD s after the time its chapter group
logged ("voice <key> at <t> s" in the manim output), the group offsets being the frame counts of the group videos; writes a
loudness-normalised AAC track, soft English subtitles (the written narration, mov_text) and <out>.srt next to the video.
usage (the environment of voice/pixi.toml: ffmpeg, numpy, soundfile):
    python mix.py <joined.mp4> <out.mp4> --voice <voice dir> --groups <g1.mp4> <g1.log> <g2.mp4> <g2.log> ..."""
import argparse
import json
import os
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf

VOICE_LEAD = 0.3  # as in ttjpsi.py
FPS = 30


def frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def cues(text, t0, dur, width=42):
    """the segment text in subtitle cues of up to two lines, timed in proportion to their length"""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    chunks = ["\n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
    n = sum(len(c) for c in chunks)
    t, out = t0, []
    for c in chunks:
        d = dur * len(c) / n
        out.append((t, t + d, c))
        t += d
    return out


ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--voice", required=True)
ap.add_argument("--groups", nargs="+", required=True)
args = ap.parse_args()
narr = json.load(open(os.path.join(args.voice, "narration.json")))
total = frames(args.video) / FPS
starts, off = {}, 0.0
for vid, log in zip(args.groups[::2], args.groups[1::2]):
    for key, t in re.findall(r"voice (\S+) at ([0-9.]+) s", open(log).read()):
        starts[key] = off + float(t)
    off += frames(vid) / FPS
assert abs(off - total) < 0.1, f"the groups add up to {off:.2f} s, the video has {total:.2f} s"
missing = [k for k in narr if k not in starts]
assert not missing, f"no start logged for {missing}"
sr = sf.info(os.path.join(args.voice, f"{next(iter(narr))}.wav")).samplerate
track = np.zeros(int((total + 1) * sr), dtype=np.float32)
subs, end_prev = [], 0.0
for key in sorted(narr, key=starts.get):
    x, sr_ = sf.read(os.path.join(args.voice, f"{key}.wav"), dtype="float32")
    assert sr_ == sr
    t0 = starts[key] + VOICE_LEAD
    assert t0 >= end_prev - 1e-3, f"{key} starts at {t0:.2f} s, before the previous segment ends ({end_prev:.2f} s)"
    i = int(round(t0 * sr))
    track[i:i + len(x)] += x
    end_prev = t0 + len(x) / sr
    subs += cues(narr[key], t0, len(x) / sr)
track = track[:int(total * sr)]
base = os.path.splitext(args.out)[0]
with open(base + ".srt", "w") as f:
    for n, (a, b, c) in enumerate(subs, 1):
        f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{c}\n\n")
with tempfile.TemporaryDirectory() as tmp:
    wav = os.path.join(tmp, "voice.wav")
    sf.write(wav, track, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.video, "-i", wav, "-i", base + ".srt", "-map", "0:v", "-map", "1:a", "-map", "2:s",
                    "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "160k",
                    "-c:s", "mov_text", "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart",
                    args.out], check=True)
print(f"{len(narr)} segments on {total:.1f} s of video, {len(subs)} subtitle cues -> {args.out}, {base}.srt")
