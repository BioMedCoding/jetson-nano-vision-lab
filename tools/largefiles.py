#!/usr/bin/env python3
"""Keep files larger than GitHub's 100 MB limit inside the repo, as chunks.

Why: this project targets an old board (Jetson Nano, JetPack 4.6.1). The
binaries it depends on (a CUDA-enabled torch wheel for Python 3.6, TensorRT
engines, ONNX exports) are exactly the kind of thing whose download links
disappear. So they live in git, split into chunks below the limit, and are
reassembled + SHA-256 verified locally.

Layout, for a large file `path/to/model.engine`:
    path/to/model.engine.part-00, .part-01, ...   committed
    path/to/model.engine                          reassembled, git-ignored
    largefiles.manifest (repo root)               path, size, sha256, parts

Usage (run from anywhere inside the repo; Python >= 3.6, stdlib only):
    tools/largefiles.py split <file> [<file> ...]   chunk + record in manifest
    tools/largefiles.py join [<prefix>]             reassemble missing/stale files
    tools/largefiles.py verify [<prefix>]           check reassembled files
    tools/largefiles.py list

`<prefix>` limits join/verify to manifest paths starting with it
(e.g. `nano/` on the Jetson, which never needs training files).
"""
import argparse
import hashlib
import os
import sys

CHUNK_BYTES = 90 * 1024 * 1024          # comfortably below GitHub's 100 MB
MANIFEST = "largefiles.manifest"
GITIGNORE_BEGIN = "# >>> largefiles (managed by tools/largefiles.py, do not edit)"
GITIGNORE_END = "# <<< largefiles"


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def sha256_of(path, extra_paths=()):
    h = hashlib.sha256()
    for p in [path] + list(extra_paths):
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def read_manifest(root):
    entries = {}
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rel, size, digest, parts = line.split()
            entries[rel] = {"size": int(size), "sha256": digest, "parts": int(parts)}
    return entries


def write_manifest(root, entries):
    with open(os.path.join(root, MANIFEST), "w") as f:
        f.write("# path  size_bytes  sha256  n_parts  -- see tools/largefiles.py\n")
        for rel in sorted(entries):
            e = entries[rel]
            f.write("{} {} {} {}\n".format(rel, e["size"], e["sha256"], e["parts"]))
    update_gitignore(root, sorted(entries))


def update_gitignore(root, rel_paths):
    path = os.path.join(root, ".gitignore")
    text = open(path).read() if os.path.exists(path) else ""
    if GITIGNORE_BEGIN in text:
        head = text[:text.index(GITIGNORE_BEGIN)]
        tail = text[text.index(GITIGNORE_END) + len(GITIGNORE_END):]
    else:
        head, tail = text.rstrip("\n") + "\n\n", "\n"
    block = [GITIGNORE_BEGIN] + ["/" + p for p in rel_paths] + [GITIGNORE_END]
    with open(path, "w") as f:
        f.write(head + "\n".join(block) + tail)


def part_paths(abs_path, n):
    return ["{}.part-{:02d}".format(abs_path, i) for i in range(n)]


def cmd_split(root, files):
    entries = read_manifest(root)
    for f in files:
        abs_path = os.path.abspath(f)
        rel = os.path.relpath(abs_path, root)
        size = os.path.getsize(abs_path)
        digest = sha256_of(abs_path)
        n = 0
        with open(abs_path, "rb") as src:
            while True:
                chunk = src.read(CHUNK_BYTES)
                if not chunk:
                    break
                with open("{}.part-{:02d}".format(abs_path, n), "wb") as out:
                    out.write(chunk)
                n += 1
        entries[rel] = {"size": size, "sha256": digest, "parts": n}
        print("split   {} -> {} parts ({:.1f} MB)".format(rel, n, size / 1e6))
    write_manifest(root, entries)


def _selected(entries, prefix):
    return [r for r in sorted(entries) if not prefix or r.startswith(prefix)]


def cmd_join(root, prefix):
    entries = read_manifest(root)
    ok = True
    for rel in _selected(entries, prefix):
        e = entries[rel]
        abs_path = os.path.join(root, rel)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) == e["size"] \
                and sha256_of(abs_path) == e["sha256"]:
            print("ok      {}".format(rel))
            continue
        parts = part_paths(abs_path, e["parts"])
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            print("MISSING {} (no {})".format(rel, os.path.relpath(missing[0], root)))
            ok = False
            continue
        if sha256_of(parts[0], parts[1:]) != e["sha256"]:
            print("CORRUPT {} (chunks do not match the manifest sha256)".format(rel))
            ok = False
            continue
        tmp = abs_path + ".joined"
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as src:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        out.write(block)
        os.replace(tmp, abs_path)
        print("joined  {}".format(rel))
    return ok


def cmd_verify(root, prefix):
    entries = read_manifest(root)
    ok = True
    for rel in _selected(entries, prefix):
        abs_path = os.path.join(root, rel)
        if not os.path.exists(abs_path):
            print("absent  {} (run: tools/largefiles.py join)".format(rel))
            ok = False
        elif sha256_of(abs_path) != entries[rel]["sha256"]:
            print("BAD     {}".format(rel))
            ok = False
        else:
            print("ok      {}".format(rel))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("split")
    p.add_argument("files", nargs="+")
    for name in ("join", "verify"):
        p = sub.add_parser(name)
        p.add_argument("prefix", nargs="?", default="")
    sub.add_parser("list")
    args = ap.parse_args()
    root = repo_root()

    if args.cmd == "split":
        cmd_split(root, args.files)
    elif args.cmd == "join":
        sys.exit(0 if cmd_join(root, args.prefix) else 1)
    elif args.cmd == "verify":
        sys.exit(0 if cmd_verify(root, args.prefix) else 1)
    elif args.cmd == "list":
        for rel, e in sorted(read_manifest(root).items()):
            print("{:>10.1f} MB  {:>2} parts  {}".format(e["size"] / 1e6, e["parts"], rel))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
