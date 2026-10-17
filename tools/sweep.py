# SPDX-License-Identifier: MIT
"""Run the whole benchmark sweep, then tabulate and plot it.

For every installed compiler the native bench (hand-derived and JAX-exported materials) on
hex8; for the first compiler also every element type and order of --elements; then, if
.venv exists, the embedded-JAX and socket benches; optionally heap allocations per element
with valgrind. Tables and the compiler and paradigm plots use hex8. Everything lands in results/<timestamp>/:
  env.json                  machine, compilers, versions, git commit, arguments
  <bench>_<compiler>.csv    raw rows of each bench (long format, see AGENTS.md)
  results.csv               all rows merged
  summary.md                tables of element and material time
  *.png                     plots

Usage: .venv/bin/python tools/sweep.py [--help]
"""
import argparse
import csv
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILERS = ["gfortran", "ifx", "nvfortran", "flang"]
MODELS = ["svk", "nh", "mnh", "mr"]
CFGS = ["ref", "cur", "mix"]
FIELDS = ["paradigm", "compiler", "model", "cfg", "element", "order", "nnodes", "npoints", "batch", "quantity", "value"]


def oneapi_env():
    """Environment with ifx on PATH (setvars.sh refuses to run twice without --force)."""
    if shutil.which("ifx"):
        return dict(os.environ)
    out = subprocess.run(["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1; env -0"],
                         capture_output=True, text=True).stdout
    return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)


def first_line(cmd, env=None):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def run(cmd, cwd=ROOT, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def environment(args, compilers, envs):
    cpu = next((l.split(":", 1)[1].strip() for l in subprocess.run(["lscpu"], capture_output=True, text=True)
                .stdout.splitlines() if l.startswith("Model name")), platform.processor())
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "host": platform.node(),
        "kernel": platform.release(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "compilers": {fc: first_line([fc, "--version"], envs[fc]) for fc in compilers},
        "fpm": first_line(["fpm", "--version"]),
        "python": sys.version.split()[0],
        "arguments": vars(args),
    }
    try:
        import jax
        info["jax"] = jax.__version__
    except ImportError:
        pass
    return info


def allocations(fc, env, out, args):
    """Heap allocations per element (element and material sweep) for each paradigm, model, cfg."""
    exe = subprocess.run(["fpm", "run", "--compiler", fc, "--profile", "release", "--runner", "echo"],
                         cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    rows = []
    for paradigm in ("native", "export"):
        for model in MODELS:
            for cfg in CFGS:
                counts, meta = [], None
                for n in (10, 20):
                    tmp = os.path.join(out, "alloc_tmp.csv")
                    res = subprocess.run(["valgrind", exe, "--nelem", str(n), "--ntrial", "1", "--models", model,
                                          "--cfgs", cfg, "--paradigms", paradigm, "--csv", tmp, "--tag", fc],
                                         cwd=ROOT, env=env, capture_output=True, text=True)
                    counts.append(int(re.search(r"total heap usage: ([\d,]+) allocs", res.stderr)
                                      .group(1).replace(",", "")))
                    meta = read_rows(tmp)[0]
                os.remove(tmp)
                rows.append({**meta, "batch": meta["npoints"], "quantity": "heap_allocs_per_element",
                             "value": (counts[1] - counts[0]) / 10})
    return rows


def hex8(rows):
    """Rows of the trilinear hexahedron, the element every bench shares."""
    return [r for r in rows if r["element"] == "hex" and r["order"] == "1"]


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def table(rows, quantity, title, unit):
    """Markdown table: one row per model and cfg, one column per paradigm and compiler."""
    sel = [r for r in rows if r["quantity"] == quantity and r["model"] in MODELS]
    if not sel:
        return ""
    cols = sorted({(r["paradigm"], r["compiler"], int(r["batch"])) for r in sel},
                  key=lambda c: (["native", "export", "jax", "socket"].index(c[0]) if c[0] in
                                 ["native", "export", "jax", "socket"] else 9, c[1], c[2]))
    val = {(r["model"], r["cfg"], r["paradigm"], r["compiler"], int(r["batch"])): float(r["value"]) for r in sel}
    npts = sel[0]["npoints"]
    head = [f"{p} {c}" + ("" if p in ("native", "export", "cached") else f" b={b}") for p, c, b in cols]
    lines = [f"### {title} ({unit})", "", "| model | cfg | " + " | ".join(head) + " |",
             "|---|---|" + "---:|" * len(cols)]
    for m in MODELS:
        for g in CFGS:
            cells = [val.get((m, g, *c)) for c in cols]
            if any(v is not None for v in cells):
                lines.append(f"| {m} | {g} | " + " | ".join("" if v is None else f"{v:.4g}" for v in cells) + " |")
    return "\n".join(lines) + "\n"


def element_table(rows):
    """Markdown table: native element time per element type and order (rows), model and cfg (columns)."""
    sel = [r for r in rows if r["paradigm"] == "native" and r["quantity"] == "element_us"]
    keys = sorted({(r["element"], int(r["order"]), int(r["nnodes"]), int(r["npoints"]), r["compiler"]) for r in sel})
    if len({k[:2] for k in keys}) < 2:
        return ""
    val = {(r["element"], int(r["order"]), r["compiler"], r["model"], r["cfg"]): float(r["value"]) for r in sel}
    comp = keys[0][4]
    lines = [f"### Element R and K by element ({comp}, native, us per element)", "",
             "| element | nodes | points | " + " | ".join(f"{m} {g}" for m in MODELS for g in CFGS) + " |",
             "|---|---:|---:|" + "---:|" * 12]
    for el, p, nn, npt, c in keys:
        if c == comp:
            lines.append(f"| {el} p={p} | {nn} | {npt} | " +
                         " | ".join(f"{val.get((el, p, c, m, g), float('nan')):.4g}" for m in MODELS for g in CFGS) + " |")
    return "\n".join(lines) + "\n"


def plots(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{m}\n{g}" for m in MODELS for g in CFGS]
    x = np.arange(len(labels))

    def series(paradigm, compiler, quantity, batch=None):
        d = {(r["model"], r["cfg"]): float(r["value"]) for r in rows
             if r["paradigm"] == paradigm and r["compiler"] == compiler and r["quantity"] == quantity
             and (batch is None or r["batch"] == str(batch))}
        return np.array([d.get((m, g), np.nan) for m in MODELS for g in CFGS])

    files = []
    elements = rows
    rows = hex8(rows)
    native = sorted({r["compiler"] for r in rows if r["paradigm"] == "native" and r["quantity"] == "element_us"
                     and r["batch"] == r["npoints"]})
    main = [c for c in native if any(r["paradigm"] == "export" and r["compiler"] == c for r in rows)]

    # 1. element time per compiler, hand-derived against JAX-exported
    if main:
        fig, axes = plt.subplots(len(main), 1, figsize=(11, 3.2 * len(main)), sharex=True, squeeze=False)
        for ax, fc in zip(axes[:, 0], main):
            for k, (p, lab) in enumerate([("native", "hand-derived"), ("export", "JAX-exported")]):
                ax.bar(x + (k - 0.5) * 0.4, series(p, fc, "element_us"), 0.4, label=lab)
            ax.set_ylabel("element R, K (us)")
            ax.set_title(fc, loc="left", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
        axes[-1, 0].set_xticks(x, labels, fontsize=8)
        fig.tight_layout()
        files.append("element_time.png")
        fig.savefig(os.path.join(out, files[-1]), dpi=130)
        plt.close(fig)

    # 2. compilers side by side, hand-derived routines
    if len(native) > 1:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        w = 0.8 / len(native)
        for k, fc in enumerate(native):
            ax.bar(x + (k - (len(native) - 1) / 2) * w, series("native", fc, "element_us"), w, label=fc)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("element R, K (us)")
        ax.set_title("hand-derived routines by compiler", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        files.append("compilers.png")
        fig.savefig(os.path.join(out, files[-1]), dpi=130)
        plt.close(fig)

    # 3. material time per point against batch size, Python paradigms against native levels
    remote = [p for p in ("jax", "socket") if any(r["paradigm"] == p for r in rows)]
    if remote:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.6), sharey=True)
        for ax, m in zip(axes, MODELS):
            for p in remote:
                for g, style in zip(CFGS, ["-", "--", ":"]):
                    pts = sorted((int(r["batch"]), float(r["value"])) for r in rows
                                 if r["paradigm"] == p and r["model"] == m and r["cfg"] == g
                                 and r["quantity"] == "material_ns_per_point")
                    if pts:
                        ax.plot(*zip(*pts), style, marker="o", ms=3, label=f"{p} {g}",
                                color="C0" if p == "jax" else "C1")
            nat = [float(r["value"]) for r in rows if r["paradigm"] == "native" and r["model"] == m
                   and r["quantity"] == "material_ns_per_point" and r["compiler"] == (main or native or ["?"])[0]]
            if nat:
                ax.axhline(max(nat), color="k", lw=0.8, ls="--", label="native (max over cfg)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("points per call")
            ax.set_title(m, loc="left", fontsize=10)
        axes[0].set_ylabel("material time (ns per point)")
        axes[-1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        files.append("batch_scaling.png")
        fig.savefig(os.path.join(out, files[-1]), dpi=130)
        plt.close(fig)
    # 4. element time against nodes per element, every element type and order
    sel = [r for r in elements if r["quantity"] == "element_us" and r["paradigm"] in ("native", "export")]
    if len({(r["element"], r["order"]) for r in sel}) > 1:
        comp = sorted({r["compiler"] for r in sel if r["element"] != "hex" or r["order"] != "1"})[0]
        fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.6), sharey=True)
        marks = {"hex": "s", "tet": "^", "wedge": "D"}
        for ax, m in zip(axes, MODELS):
            for g, color in zip(CFGS, ["C0", "C1", "C2"]):
                for p, fill in (("native", True), ("export", False)):
                    pts = sorted((int(r["nnodes"]), float(r["value"]), r["element"]) for r in sel
                                 if r["compiler"] == comp and r["model"] == m and r["cfg"] == g and r["paradigm"] == p)
                    for el in marks:
                        xy = [(n, v) for n, v, e in pts if e == el]
                        if xy:
                            ax.plot(*zip(*xy), marker=marks[el], ls="-" if p == "native" else ":", color=color,
                                    mfc=color if fill else "none", ms=5,
                                    label=f"{g} {p}" if el == "hex" else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("nodes per element")
            ax.set_title(m, loc="left", fontsize=10)
        axes[0].set_ylabel("element R, K (us)")
        axes[-1].legend(frameon=False, fontsize=7)
        fig.suptitle(f"{comp}: square hex, triangle tet, diamond wedge; filled hand-derived, open JAX-exported",
                     fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        files.append("element_order.png")
        fig.savefig(os.path.join(out, files[-1]), dpi=130)
        plt.close(fig)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nelem", type=int, default=2000, help="elements for the native bench (default 2000)")
    ap.add_argument("--ntrial", type=int, default=7, help="timings per measurement (default 7)")
    ap.add_argument("--python-nelem", type=int, default=200, help="elements for the jax and socket benches")
    ap.add_argument("--compilers", default=",".join(COMPILERS), help="comma-separated, default: all installed")
    ap.add_argument("--elements", default="hex:1,hex:2,tet:1,tet:2,wedge:1,wedge:2",
                    help="element:order list run with the first compiler (default %(default)s)")
    ap.add_argument("--no-python", action="store_true", help="skip the jax and socket benches")
    ap.add_argument("--allocations", action="store_true", help="heap allocations per element with valgrind")
    ap.add_argument("--out", help="output directory (default results/<timestamp>)")
    args = ap.parse_args()

    out = os.path.abspath(args.out or os.path.join(ROOT, "results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    ienv = oneapi_env()
    envs = {fc: (ienv if fc == "ifx" else dict(os.environ)) for fc in args.compilers.split(",")}
    compilers = [fc for fc in args.compilers.split(",") if shutil.which(fc, path=envs[fc].get("PATH"))]
    with open(os.path.join(out, "env.json"), "w") as fh:
        json.dump(environment(args, compilers, envs), fh, indent=2)

    rows = []
    for fc in compilers:
        f = os.path.join(out, f"native_{fc}.csv")
        run(["fpm", "run", "--compiler", fc, "--profile", "release", "--", "--nelem", str(args.nelem),
             "--ntrial", str(args.ntrial), "--csv", f, "--tag", fc], env=envs[fc])
        rows += read_rows(f)
        if args.allocations and shutil.which("valgrind"):
            rows += allocations(fc, envs[fc], out, args)
    for spec in args.elements.split(",") if compilers else []:
        el, order = spec.split(":")
        if (el, order) == ("hex", "1"):
            continue
        fc = compilers[0]
        f = os.path.join(out, f"native_{fc}_{el}{order}.csv")
        run(["fpm", "run", "--compiler", fc, "--profile", "release", "--", "--nelem", str(args.nelem),
             "--ntrial", str(args.ntrial), "--element", el, "--order", order, "--csv", f, "--tag", fc], env=envs[fc])
        rows += read_rows(f)
    python = os.path.join(ROOT, ".venv", "bin", "python")
    if not args.no_python and os.path.exists(python):
        common = ["--nelem", str(args.python_nelem), "--ntrial", str(args.ntrial), "--tag", "gfortran"]
        f = os.path.join(out, "jax_gfortran.csv")
        run([os.path.join(ROOT, "jax", "run.sh"), "run", "--profile", "release", "--", *common, "--csv", f])
        rows += read_rows(f)
        f = os.path.join(out, "socket_gfortran.csv")
        run(["fpm", "run", "--profile", "release", "--", *common, "--csv", f], cwd=os.path.join(ROOT, "socket"))
        rows += read_rows(f)

    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)
    summary = "\n".join(t for t in [
        table(hex8(rows), "element_us", "Element R and K, hex8", "us per element"),
        table(hex8(rows), "material_ns_per_point", "Material call, hex8", "ns per point"),
        table(hex8(rows), "heap_allocs_per_element", "Heap allocations, hex8", "per element, element and material sweep"),
        element_table(rows),
    ] if t)
    files = plots(rows, out)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(f"# Sweep {os.path.basename(out)}\n\nSee env.json for the machine and versions.\n\n")
        fh.write(summary + "\n" + "".join(f"![{f}]({f})\n" for f in files))
    print("\n" + summary)
    print("wrote", out)


if __name__ == "__main__":
    main()
