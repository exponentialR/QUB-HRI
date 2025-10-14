#!/usr/bin/env python3
import argparse, json, os, re, subprocess, sys, csv

def run(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8")

def parse_name(stem):
    # expects names like: p01-CAM_AV-BIAH_RB-BHO-...
    t = stem.split("-")
    if len(t) < 4: return None
    pid = t[0].upper()                  # p01 -> P01
    view = t[1]                         # CAM_AV / CAM_UL / ...
    task_actor = t[2]                   # e.g., BIAH_RB
    subtask = t[3]                      # e.g., BHO
    if "_" in task_actor:
        task, actor = task_actor.split("_", 1)
    else:
        task, actor = task_actor, ""
    return {"pid":pid,"view":view,"task":task,"actor":actor,"subtask":subtask}

def load_consented(path):
    if not path: return None
    with open(path) as f:
        return set(x.strip().upper() for x in f if x.strip())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True, help='e.g., gdrive:/QUB-PHEO-DATASET')
    ap.add_argument("--out", default="MANIFEST.tsv")
    ap.add_argument("--exts", default="mp4,mov,avi")
    ap.add_argument("--consented", default="")
    args = ap.parse_args()

    exts = set(x.strip().lower() for x in re.split(r"[,\s]+", args.exts) if x.strip())
    consented = load_consented(args.consented)

    # Get recursive JSON listing (includes MD5) from Drive root
    js = run(["rclone","lsjson",args.remote,"--recursive","--hash"])
    items = json.loads(js)

    rows = [["pid","view","task","actor","subtask","size_bytes","md5","relpath"]]
    for it in items:
        if it.get("IsDir"):
            continue
        rel = it["Path"]                               # path relative to dataset root on Drive
        name = os.path.basename(rel)
        stem, ext = os.path.splitext(name)
        if ext.lower().lstrip(".") not in exts:
            continue
        meta = parse_name(stem)
        if not meta:
            continue
        if consented and meta["pid"] not in consented:
            continue

        md5  = (it.get("Hashes") or {}).get("md5","")
        size = int(it.get("Size", 0))
        rows.append([meta["pid"],meta["view"],meta["task"],meta["actor"],meta["subtask"],size,md5,rel])

    with open(args.out,"w",newline="") as fo:
        csv.writer(fo, delimiter="\t").writerows(rows)
    print(f"Wrote {args.out} with {len(rows)-1} rows")
