#!/usr/bin/env python3
"""Cache the minimum-loop public assets on lab-pc, then package them for SCP."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\cjl\parameter-importance-assets")
PAYLOAD = ROOT / "payload"
MIRROR = "https://hf-mirror.com"
PYTHIA_SOURCE_COMMIT = "a19eecb807ec2c79a39ebf18108816e6ffffc1d5"

SPECS = [
    ("model", "EleutherAI/pythia-160m-deduped", "0450b9dc64e8a49a87c9eec8813d0e6979643420", "models/pythia-160m-deduped-step0", "model"),
    ("model", "EleutherAI/pythia-160m-deduped", "eaa6383edf563bf355d8b125777f352c2a773bfa", "models/pythia-160m-deduped-step512", "model"),
    ("model", "EleutherAI/pythia-14m", "step0", "models/pythia-14m-step0", "model"),
    ("dataset", "nyu-mll/glue", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "datasets/glue-sst2", "sst2"),
    ("dataset", "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "datasets/wikitext-103-raw-v1", "wikitext-103-raw-v1"),
]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "parameter-importance-assets/1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_files(siblings: list[dict], selector: str) -> list[dict]:
    if selector == "model":
        allowed = {
            "config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.json", "merges.txt",
        }
        result = [x for x in siblings if x["rfilename"] in allowed]
        if "model.safetensors" not in {x["rfilename"] for x in result}:
            raise RuntimeError("revision has no model.safetensors")
        if "config.json" not in {x["rfilename"] for x in result}:
            raise RuntimeError("revision has no config.json")
        return result
    prefix = selector + "/"
    result = [x for x in siblings if x["rfilename"].startswith(prefix)]
    if not result:
        raise RuntimeError(f"no files matched {prefix}")
    return result


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    part = destination.with_name(destination.name + ".part")
    subprocess.run([
        "curl.exe", "-q", "--noproxy", "*", "--fail", "--location",
        "--retry", "5", "--retry-delay", "3", "--continue-at", "-",
        "--output", str(part), url,
    ], check=True)
    os.replace(part, destination)


def main() -> None:
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    resolved: list[dict] = []
    for repo_type, repo, requested_revision, relative_dir, selector in SPECS:
        api_kind = "models" if repo_type == "model" else "datasets"
        api = f"{MIRROR}/api/{api_kind}/{repo}/revision/{requested_revision}?blobs=true"
        info = get_json(api)
        revision = info["sha"]
        if len(revision) != 40:
            raise RuntimeError(f"API did not resolve a full commit for {repo}@{requested_revision}")
        files = selected_files(info["siblings"], selector)
        resolved.append({"type": repo_type, "repo": repo, "requested_revision": requested_revision, "revision": revision, "files": [x["rfilename"] for x in files]})
        prefix = "" if repo_type == "model" else "datasets/"
        for sibling in files:
            name = sibling["rfilename"]
            quoted = urllib.parse.quote(name, safe="/")
            url = f"{MIRROR}/{prefix}{repo}/resolve/{revision}/{quoted}"
            # Dataset config prefixes are removed because each local directory is single-config.
            local_name = name if selector == "model" else name.split("/", 1)[1]
            destination = PAYLOAD / relative_dir / local_name
            download(url, destination)
            record = {
                "repo_type": repo_type,
                "repo": repo,
                "revision": revision,
                "source_file": name,
                "local_file": str(destination.relative_to(PAYLOAD)).replace("\\", "/"),
                "size": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
            lfs = sibling.get("lfs") or {}
            upstream = lfs.get("oid", "")
            if upstream.startswith("sha256:") and record["sha256"] != upstream[7:]:
                raise RuntimeError(f"upstream SHA-256 mismatch: {name}")
            records.append(record)

    # Pin and archive the official source snapshot used for data-layout reference.
    commit = PYTHIA_SOURCE_COMMIT
    source = PAYLOAD / "source" / f"pythia-{commit}.tar.gz"
    download(f"https://codeload.github.com/EleutherAI/pythia/tar.gz/{commit}", source)
    records.append({
        "repo_type": "git", "repo": "EleutherAI/pythia", "revision": commit,
        "source_file": f"{commit}.tar.gz", "local_file": str(source.relative_to(PAYLOAD)).replace("\\", "/"),
        "size": source.stat().st_size, "sha256": file_sha256(source),
    })

    manifests = PAYLOAD / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / "asset-revisions.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (manifests / "asset-manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with (manifests / "asset-sha256.tsv").open("w", encoding="ascii", newline="\n") as f:
        for item in sorted(records, key=lambda x: x["local_file"]):
            f.write(f'{item["sha256"]}\t{item["size"]}\t{item["local_file"]}\n')

    archive = ROOT / "minimum-loop-assets.tar"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w") as tf:
        for path in PAYLOAD.iterdir():
            tf.add(path, arcname=path.name)
    print(json.dumps({"archive": str(archive), "size": archive.stat().st_size, "files": len(records)}))


if __name__ == "__main__":
    main()
