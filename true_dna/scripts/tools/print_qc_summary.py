import json
import os

p = "qc_results/summary.json"
if not os.path.exists(p):
    print("summary.json not found at", p)
    raise SystemExit(1)

with open(p) as f:
    s = json.load(f)


def pct(x, total):
    return 100.0 * x / total if total else 0.0


print("QC Summary:")
print("------------")

global_stats = s.get("global_stats", {})
global_counts = s.get("global", {}).get("base_counts", {})

total_seqs = s.get("global", {}).get("total_sequences", 0)
removed = s.get("global", {}).get("removed_sequences", 0)
print(f"Total sequences: {total_seqs}")
print(f"Removed sequences (>N threshold): {removed} ({pct(removed, total_seqs):.2f}%)\n")

# GC stats
gs = global_stats.get("gc_percents", {})
if gs.get("count", 0):
    print("GC%: count={count} min={min:.2f} max={max:.2f} mean={mean:.2f} median={median:.2f}".format(**gs))
else:
    print("GC%: no data")

# N% stats
ns = global_stats.get("n_percents", {})
if ns.get("count", 0):
    print("N%: count={count} min={min:.2f} max={max:.2f} mean={mean:.2f} median={median:.2f}\n".format(**ns))
else:
    print("N%: no data")

# Length stats
ls = global_stats.get("lengths", {})
if ls.get("count", 0):
    print("Sequence lengths: count={count} min={min} max={max} mean={mean:.1f} median={median}".format(**ls))
    hist = ls.get("histogram", [])
    if hist:
        print(f"Length histogram ({len(hist)} bins):")
        print(", ".join(str(x) for x in hist))
else:
    print("Lengths: no data")

# base composition
print("\nBase composition (global counts and percentages):")
base_counts = global_counts
total_bases = sum(base_counts.values())
for b in sorted(base_counts.keys()):
    cnt = base_counts[b]
    print(f"  {b}: {cnt} ({pct(cnt, total_bases):.2f}%)")

print("\nPer-file removed summary:")
for fname, info in s.get("files", {}).items():
    tot = info.get("total_sequences", 0)
    rem = info.get("removed_sequences", 0)
    print(f"  {fname}: total={tot}, removed={rem}, percent_removed={pct(rem, tot):.2f}%")

print("\nFiltered FASTA files written to qc_results/filtered_<basename>")
