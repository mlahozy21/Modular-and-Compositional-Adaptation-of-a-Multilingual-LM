# Does adaptation compose?

Independently trained language and genre modules on a low-resource intersection
domain. ELMA/SILD take-home, CAIDAS, JMU Würzburg.

A text corpus sits at the intersection of several dimensions at once — the
language it is written in, the period, the genre — and the domains that matter
for the humanities are the cells where almost no text exists, while the margins
that contain them are data-rich. This study asks whether adaptation to such a
cell can be **assembled from modules trained independently on those margins**,
for a language factor and a genre factor, on the cell ⟨Swedish, literary⟩.

**The headline.** Neither module helps the target on its own — each costs about
`+0.023` bpb and their plain sum costs `+0.038`. Averaged, they help: `−0.010`
bpb with no target data at all, in all five test chapters. One fitted scalar per
module takes that to `−0.016`, of which the genre factor at a fitted scale
accounts for roughly seven tenths and the language factor for the rest. What
governs the composition is the **magnitude** of the two updates and not their
**direction**: three probes across two different spaces recover at most a tenth
of what scaling delivers.

The paper is in [`paper/paper.pdf`](paper/paper.pdf).

---

## Layout

```
paper/
  paper.pdf                 the report
  paper.tex                 its source (XeLaTeX; needs pgfplots, no external .bib)
notebook/
  elma_sild_sv.ipynb        the whole study, end to end — start here
  elma_sild_fi.ipynb        the same protocol on ⟨fi, literary⟩, not in the paper
data/
  manifest_sv.json          sizes and sha256 of the two module corpora as built
  manifest_fi.json          the same for the Finnish run
  corpus_counts/            corpus-level measurements cited in the paper,
                            including the chapter split of each target
scripts/
  measure_fertility.py      tokeniser fertility on the target cell (no GPU)
requirements.txt            version constraints, and why they matter here
LICENSE                     MIT for the code; the corpora are not redistributed
```

There is no `raw/` directory in the repository. Both datasets are pulled from the
HuggingFace Hub by the notebooks themselves; nothing is redistributed here.

---

## How to read the notebook

Every code cell carries one of two banners, and the split is the fastest way in.

**`══ PLUMBING ══`** — infrastructure. Micro-batching and OOM handling, mounting
and unmounting adapters, tokenising, subsampling hidden states, writing JSON.
None of it is part of the argument: any of these cells could be replaced by a
different implementation without changing a single claim. Ten cells.

**`── CORE ──`** — the study. Every design decision, every measurement and every
result. If a number appears in the paper, a CORE cell prints it. Twenty-two
cells.

The notebook runs top to bottom in Colab on a single A100. Measured: **2.9 h**
for the two trainings — 35 min for the language module, 136 min for the genre
one — and about an hour of evaluation. Trained adapters are cached to Drive as
`tau_language_r16.pt` and `tau_genre_r16.pt`, so a re-run skips the training and
only re-measures.

**A second cell, not reported in the paper.** `elma_sild_fi.ipynb` runs the
identical protocol on ⟨Finnish, literary⟩ — same banners, same cell count, same
code path, only the target changed. The paper reports ⟨Swedish, literary⟩ only;
this run is here because it exists and because the question of what survives a
change of cell is the obvious next one, not because anything in the report
depends on it. Its CORE cells print the same quantities for the Finnish cell, so
the two runs can be compared by anyone who wants to.

Note that `requirements.txt` governs the local script path. The notebook
installs its own dependencies with `pip -U` at the top, because a Colab runtime
ships older versions; what it resolves to is printed in section 2 and recorded
in the exported summary.

---

## What the study does

Three trainings and two evaluation settings, with `l = sv` and `g = literary`.

| module | trained on | sees the target? |
|---|---|---|
| `τ_L` language | ⟨sv, ¬literary⟩ — `opus_dgt`, pair `hr-sv` | no |
| `τ_G` genre | ⟨L∖sv, literary⟩ — `opus_books` minus every pair containing `sv` | no |
| composition | the two above, combined | evaluated on the target test split |

Two decisions in preparing the data are not cosmetic, and the notebook makes
both explicit.

**Training blocks must be language-pure.** `opus_books` stores translation
*pairs*, so chunking rows as they come out of the parquet files yields blocks
that alternate language every segment, with the same sentence appearing twice.
That trains translation, not genre. Blocks are chunked *within* each language and
only the resulting list is shuffled.

**The target is split by chapter.** The Swedish literary slice is one work in
three segmentations, so cutting by row or by pair would put the same sentence on
both sides. All three copies of a chapter go to the same partition, and the
paired bootstrap resamples **chapters**, not blocks, for the same reason.

---

## Implementation details

Everything needed to reproduce the runs. All of it is set in section 1 of the
notebook.

| | |
|---|---|
| backbone | `BSC-LT/salamandra-2b`, 2,253,490,176 parameters, `bfloat16`, frozen |
| adapter | LoRA, `r = 16`, `α = 32` (so `α/r = 2`), dropout 0, no bias |
| adapted matrices | `q,k,v,o,gate,up,down_proj` of all 24 blocks — 168 matrices |
| trainable parameters | 14,917,632 — **0.66 %** of the 2.25B total, **1.24 %** of the 1.20B outside the embeddings; promoted to fp32 |
| optimiser | AdamW, `lr = 1e-4`, `betas = (0.9, 0.95)`, `weight_decay = 0` |
| schedule | OneCycleLR, `max_lr = 1e-4`, `pct_start = 0.03`, cosine anneal |
| gradient clipping | norm 1.0 |
| block length | 512 tokens (511 predicted) |
| batch | 16 blocks = 8,192 tokens per optimisation step |
| micro-batch | 8 blocks, halved automatically on OOM (2 for the per-layer fit) |
| passes | **exactly one** over each module corpus, no repetition |
| steps | 2,520 for `τ_L`, 9,810 for `τ_G` |
| seed | 20260816, for data order, shuffling and torch |
| held out | 96 blocks from the end of each module corpus, only to measure it |
| hardware | one A100 40 GB (Colab) |

**Supervised stage.** The 1M-word budget is spent on **two scalars**, estimated
by exhaustive grid search of step 0.2 over `[0,1]²`, refined to 0.1 in the 3×3
neighbourhood of the winner, scored on the *whole* training split of the target
(366 blocks, 753 kB). Validation and test are never consulted for the choice. A
per-layer variant fits 48 scalars by gradient descent from the global optimum;
it is reported as exploratory.

**Statistics.** Paired bootstrap, 10,000 resamples, resampling chapters. With
five chapters the percentile interval is essentially the range of the five
chapter differences, so those are quoted alongside it.

---

## Where each number comes from

| in the paper | notebook section |
|---|---|
| corpus sizes and the target split (§2) | 3 · The corpora |
| fertility, backbone selection (§3) | `scripts/measure_fertility.py` |
| each module on its own domain, Table 1 | 6 |
| subspace overlap, Table 2 | 8 |
| zero-shot systems and CIs, Table 3 | 7 and 10 |
| two fitted scalars, Table 4 | 11 |
| per-layer depth profile, Figure 1 | 12 |
| the scaled single-module control, the 30/70 split | 13 |
| per-token profile (§8.1) | 14 |
| activation alignment, axis projections (§8.2) | 15 |
| confound axes and the intervention (§8.2) | 16 |

Running a notebook writes one JSON per stage next to the checkpoints
(`lambdas.json`, `lambdas_per_layer.json`, `single_module_control.json`,
`gain_profile.json`, `representations.json`, `confound_axes.json`) plus a summary
named after its cell, `elma_sild_sv.json` or `elma_sild_fi.json`.

---

## Reproducing the corpus counts

The corpus-level figures in `data/corpus_counts/` are cited in the paper and can
be recomputed without a GPU. The script fetches `opus_books` from the Hub itself
if `raw/` is not there, so this works from a clean checkout:

```bash
pip install -r requirements.txt
python scripts/measure_fertility.py       # writes fertility.json
```

`requirements.txt` carries floors rather than exact pins, and says so: the run
reported in the paper installed with `pip install -U` and did not capture what
it resolved to. Section 2 of the notebook now prints the resolved versions and
the export cell records them in `elma_sild_sv.json`, so any future run is
self-documenting. This is not a formality — the file lists four APIs the study
depends on that have moved between releases, three of which fail silently.

The rest — per-language word counts, each target's chapter split, and the sha256
of the module corpora in `manifest_sv.json` and `manifest_fi.json` — are produced
by section 3 of the corresponding notebook, which is the same code path.

---

## What is not here

- **The trained adapters.** Two `.pt` files of ~60 MB per cell, four in all; each
  notebook regenerates its own from the seed.
- **The raw corpora.** Downloaded from the Hub at run time.
- **A second seed.** Systems are compared on identical text, so the paired
  measurement is precise, but a single seed does not bound training variance.
  This is stated in the paper's limitations.
- **A direct-adaptation reference.** What a LoRA trained straight on the target's
  own 128k-word training split achieves is the number a practitioner would want
  first; the paper explains why it is not used as a normaliser, and why it is
  still missing.
- **A lock file.** See `requirements.txt`: the resolved versions of the reported
  run were not captured, and inventing pins after the fact would be worse than
  saying so.

---

## Data

- `Helsinki-NLP/opus_books` — out-of-copyright novels and their translations.
- `Helsinki-NLP/opus_dgt` — the Translation Memory of the European Commission's
  Directorate-General for Translation.

Both are used as published on the HuggingFace Hub, unmodified. Every corpus in
this study is translated text, which the paper names as its own most consequential
confound.
