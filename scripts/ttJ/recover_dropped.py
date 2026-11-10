"""Find the input files that finished jobs dropped silently, and write recovery sub-jsons for them.

runner.py checks every input file before processing (validate_dataset_structure): a file whose uproot.open fails, e.g.
"XRootD error: [ERROR] Operation expired" when many jobs start at once, is dropped with an "ERROR validating file" line
and the job still exits 0 with an output. This reads the attempt that wrote each output (the last "Running with the
following options" block of <submit>/out/<name>.out), lists the dropped files and, with --write, writes
<rec subjson dir>/<name>.json with only those files (root://eoscms.cern.ch where the file is on /eos/cms, else the
original URL) and <rec submit dir>/jobs.txt with the lines of <submit>/jobs.txt. Submit a copy of submit.jdl with
TTJ_SUBJSON / TTJ_OUT / initialdir pointing at the rec directories: the recovered outputs keep the job names.

usage: python3 recover_dropped.py <submit dir> <out dir> <subjson dir> [--write <rec subjson dir> <rec submit dir>]
"""
import collections
import glob
import json
import os
import re
import sys


def dropped(submit, out):
    """{job name: [dropped LFNs]} for the jobs with an output"""
    res = {}
    for f in sorted(glob.glob(f"{submit}/out/*.out")):
        name = os.path.basename(f)[:-4]
        if not os.path.exists(f"{out}/{name}/{name}.coffea"):
            continue
        last = open(f, errors="replace").read().split("Running with the following options")[-1]
        bad = sorted({"/store" + u.split("/store", 1)[1] for u in re.findall(r"ERROR validating file: (\S+?),", last)})
        if bad:
            res[name] = bad
    return res


def main():
    submit, out, subjson = sys.argv[1:4]
    res = dropped(submit, out)
    print(f"{len(res)} jobs dropped {sum(map(len, res.values()))} files")
    if "--write" not in sys.argv:
        for name, lfns in res.items():
            print(name, len(lfns))
        return
    rec_json, rec_submit = sys.argv[sys.argv.index("--write") + 1:][:2]
    os.makedirs(rec_json, exist_ok=True)
    os.makedirs(rec_submit, exist_ok=True)
    jobs = {line.split(",")[0]: line.strip() for line in open(f"{submit}/jobs.txt") if line.strip()}
    hosts = collections.Counter()
    for name, lfns in res.items():
        (key, urls), = json.load(open(f"{subjson}/{name}.json")).items()
        by_lfn = {"/store" + u.split("/store", 1)[1]: u for u in urls}
        new = []
        for lfn in lfns:
            on_eos = os.path.exists(f"/eos/cms{lfn}") and os.path.getsize(f"/eos/cms{lfn}") > 0
            new.append(f"root://eoscms.cern.ch//eos/cms{lfn}" if on_eos else by_lfn[lfn])
            hosts[new[-1].split("//")[1].split("/")[0]] += 1
        json.dump({key: new}, open(f"{rec_json}/{name}.json", "w"), indent=1)
    open(f"{rec_submit}/jobs.txt", "w").write("".join(jobs[n] + "\n" for n in res))
    print("written:", dict(hosts))


if __name__ == "__main__":
    main()
