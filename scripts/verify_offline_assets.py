#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "NO_PROXY": "*",
})

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("DATA_ROOT", "/home/sophgo13/cjl/storage/parameter-importance"))


def parquet_split(directory: Path, stem: str) -> list[str]:
    files = sorted(str(p) for p in directory.glob(f"{stem}*.parquet"))
    if not files:
        raise RuntimeError(f"no {stem} parquet files in {directory}")
    return files


def main() -> None:
    step0 = ROOT / "models/pythia-160m-deduped-step0"
    step512 = ROOT / "models/pythia-160m-deduped-step512"
    tiny = ROOT / "models/pythia-14m-step0"
    tokenizer = AutoTokenizer.from_pretrained(step0, local_files_only=True)
    verbalizers = {word: tokenizer.encode(word, add_special_tokens=False) for word in (" negative", " positive")}
    if any(len(ids) != 1 for ids in verbalizers.values()):
        raise RuntimeError(f"verbalizers are not single-token: {verbalizers}")

    model = AutoModelForCausalLM.from_pretrained(step0, local_files_only=True, torch_dtype=torch.bfloat16).cuda()
    ids = torch.tensor([[tokenizer.eos_token_id, tokenizer.eos_token_id]], device="cuda")
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("160M BF16 forward/backward produced non-finite loss")
    del model
    torch.cuda.empty_cache()
    AutoModelForCausalLM.from_pretrained(step512, local_files_only=True, torch_dtype=torch.bfloat16)
    AutoModelForCausalLM.from_pretrained(tiny, local_files_only=True, torch_dtype=torch.bfloat16)

    sst = ROOT / "datasets/glue-sst2"
    sst_data = load_dataset("parquet", data_files={
        "train": parquet_split(sst, "train"),
        "validation": parquet_split(sst, "validation"),
        "test": parquet_split(sst, "test"),
    })
    wiki = ROOT / "datasets/wikitext-103-raw-v1"
    wiki_data = load_dataset("parquet", data_files={
        "train": parquet_split(wiki, "train"),
        "validation": parquet_split(wiki, "validation"),
        "test": parquet_split(wiki, "test"),
    })
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "verbalizers": verbalizers,
        "sst2_rows": {k: v.num_rows for k, v in sst_data.items()},
        "wikitext_rows": {k: v.num_rows for k, v in wiki_data.items()},
        "status": "ok",
    }
    out = ROOT / "manifests/offline-smoke.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
