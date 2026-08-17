#!/usr/bin/env python3
"""Tokeniser fertility on the target cell, for the backbone-selection argument.

Fertility (tokens per whitespace-separated word) is a property of the pair
(tokeniser, corpus), so published figures do not transfer and it has to be
measured on the text the study actually uses. This script measures it on the
Swedish and Finnish literary slices of `Helsinki-NLP/opus_books` for the three
candidate backbones the paper compares.

Writes data/corpus_counts/fertility.json. Needs no GPU.
"""
import glob, json, os
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

RAW  = os.environ.get("RAW", "raw")
OUT  = "data/corpus_counts/fertility.json"
CANDIDATES = {"Salamandra-2b":  "BSC-LT/salamandra-2b",
              "EuroLLM-1.7B":   "utter-project/EuroLLM-1.7B",
              "Qwen3-0.6B":     "Qwen/Qwen3-0.6B"}
LANGS = ("sv", "fi")


def fetch():
    """opus_books is not redistributed here; pull it from the Hub if absent.
    Only the parquet files are downloaded, and only once — `snapshot_download`
    is a no-op when the local copy is already complete."""
    if not glob.glob(f"{RAW}/opus_books/*-*"):
        print(f"{RAW}/opus_books not found; downloading from the Hub…")
    snapshot_download("Helsinki-NLP/opus_books", repo_type="dataset",
                      local_dir=f"{RAW}/opus_books",
                      allow_patterns=["*.parquet"], max_workers=8)


def literary_slice(lang):
    """Every sentence of `lang` in opus_books, across all pairs containing it."""
    out = []
    for pair in sorted(os.path.basename(p) for p in glob.glob(f"{RAW}/opus_books/*-*")):
        if lang not in pair.split("-"):
            continue
        for f in sorted(glob.glob(f"{RAW}/opus_books/{pair}/*.parquet")):
            for row in pq.read_table(f).column("translation").to_pylist():
                s = row.get(lang) or ""
                if s.strip():
                    out.append(s)
    return out


def main():
    fetch()
    slices = {lg: literary_slice(lg) for lg in LANGS}
    for lg, segs in slices.items():
        print(f"{lg}: {len(segs):,} segments · "
              f"{sum(len(s.split()) for s in segs):,} words")

    result = {}
    for name, repo in CANDIDATES.items():
        tok = AutoTokenizer.from_pretrained(repo)
        entry = {"vocab": len(tok)}
        for lg, segs in slices.items():
            words  = sum(len(s.split()) for s in segs)
            tokens = sum(len(e) for e in
                         tok(segs, add_special_tokens=False)["input_ids"])
            entry[lg] = tokens / words
            print(f"  {name:<14} {lg}: {tokens/words:.4f} tokens per word")
        result[name] = entry

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
