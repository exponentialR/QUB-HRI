#!/usr/bin/env python3
import argparse, csv, hashlib, json, os, re, subprocess

def run_json(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return json.loads(out)

def ffprobe(path):
    j = run_json(["ffprobe","-v","error","-print_format","json","-show_streams","-show_format",path])
    v = next(s for s in j["streams"] if s.get("codec_type")=="video")
    n,d = (v.get("r_frame_rate","0/1").split("/")+["1"])[:2]
    fps = float(n)/float(d) if float(d)!=0 else 0.0
    return round(fps,3), float(j["format"].get("duration",0.0)), int(v.get("width",0)), int(v.get("height",0))

def sha256(path, buf=1024*1024):
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_name(stem):
    t = stem.split("-")
    if len(t) < 4: return None
    pid = t[0].upper(); view=t[1]; ta=t[2]; sub=t[3]
    task,actor = (ta.split("_",1)+[""])[:2] if "_" in ta else (ta,"")
    return {"pid":pid,"view":view,"task":task,"actor":actor,"subtask":sub}

def load_consented(path):
    if not path: return None
    with open(path) as f:
        return set(x.strip().upper() for x in f if x.strip())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Local/mounted dataset root (same structure as Drive)")
    ap.add_argument("--out", default="MANIFEST.tsv")
    ap.add_argument("--exts", default="mp4,mov,avi")
    ap.add_argument("--consented", default="")
    args=ap.parse_args()

    exts=set(x.strip().lower() for x in re.split(r"[,\s]+", args.exts) if x.strip())
    consented=load_consented(args.consented)

    hdr=["pid","view","task","actor","subtask","fps","duration_s","width","height","sha256","relpath"]
    rows=[hdr]
    for dp,_,fs in os.walk(args.root):
        for f in fs:
            if f.split(".")[-1].lower() not in exts: continue
            abspath=os.path.join(dp,f)
            rel=os.path.relpath(abspath, args.root)
            meta=parse_name(os.path.splitext(f)[0])
            if not meta: continue
            if consented and meta["pid"] not in consented: continue
            try:
                fps,dur,w,h=ffprobe(abspath)
            except Exception:
                fps,dur,w,h=(0.0,0.0,0,0)
            rows.append([meta["pid"],meta["view"],meta["task"],meta["actor"],meta["subtask"],
                         fps,round(dur,3),w,h,sha256(abspath),rel])
    with open(args.out,"w",newline="") as fo:
        csv.writer(fo, delimiter="\t").writerows(rows)
    print(f"Wrote {args.out} with {len(rows)-1} rows")
