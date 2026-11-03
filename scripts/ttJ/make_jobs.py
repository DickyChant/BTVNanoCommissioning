"""Split fileset JSONs ({dataset: [files]}, as written by scripts/fetch.py) into sub-jsons of N files and write the
job list of submit.jdl.template ("<name>,<isSyst>" per line).
usage: python3 make_jobs.py <subjson dir> <jobs.txt> <files per job> <fileset.json> [<fileset.json> ...] [--syst all|False]"""
import json
import os
import re
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
syst = "all" if "--syst" in sys.argv and sys.argv[sys.argv.index("--syst") + 1] == "all" else "False"
args = [a for a in args if a not in ("all", "False")]
if len(args) < 4:
    sys.exit(__doc__)
subdir, jobs, nper, filesets = args[0], args[1], int(args[2]), args[3:]
os.makedirs(subdir, exist_ok=True)
lines = []
for fs in filesets:
    for ds, files in json.load(open(fs)).items():
        tag = re.sub(r"[^A-Za-z0-9]", "", ds)
        for k in range(0, len(files), nper):
            name = f"{tag}_p{k // nper}"
            json.dump({ds: files[k : k + nper]}, open(os.path.join(subdir, name + ".json"), "w"), indent=1)
            lines.append(f"{name},{syst}")
open(jobs, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} jobs -> {jobs}, sub-jsons in {subdir}")
