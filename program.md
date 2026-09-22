# Autoresearch operating contract

The objective is to lower the fixed `score` reported by `fixed_benchmark.py`.

```
score = bpb + 2.0 * (1 - mean_benchmark_acc) + 1e-9 * parameter_count
```

`bpb` is held-out FineWeb bits-per-byte. `mean_benchmark_acc` averages truncated
MMLU, ARC-Challenge, and OpenBookQA multiple-choice accuracy scored by
log-likelihood. Lower is better.

## Literature-driven methodology

Research ideas come from top-venue papers (ICLR, NeurIPS, EMNLP, AAAI, ACL,
ICML, …), not from unlogged intuition.

1. Search or open a paper (`papers.py search|fetch|next`).
2. If the arXiv id is already in `papers.tsv`, do **not** re-read or re-try it
   unless status is being explicitly revisited with new notes.
3. Log every referred paper before using it (`papers.py log` / `search --log-new`).
4. Extract one intelligible mutation. That can be a hyperparameter, or a real
   architecture or method change in `model.py` (attention, feed-forward, norms,
   positions, initialization) drawn from a logged paper — Kimi, Qwen, Mistral,
   and the rest of the literature are in scope.
5. Run the loop from the current best checkpoint when the weights still match.
   Architecture edits train from scratch for that trial.
6. Mark the paper `applied`, `skipped`, or `rejected` with the run id and idea.

`papers.tsv` is append-only. Latest row per `arxiv_id` is the current status.

## Checkpoints

- Each evaluator run writes weights under `artifacts/<run_id>/`.
- Accepted improvements are promoted to `checkpoints/best/` (`model.pt`,
  `candidate.json`, `metrics.json`, `meta.json`).
- Resume later with:

```powershell
python .\loop.py --resume --paper-id 2406.17557 --iterations 4 --budget-seconds 30
```

Architecture-changing mutations skip weight load and train from scratch; optimizer
or schedule mutations warm-start from `checkpoints/best`.

## Files and boundaries

- `model.py` is the architecture surface. The Grok proposer rewrites it, one
  idea per trial, then the loop keeps or restores the file.
- `candidate.json` is the size and optimizer surface used by `--proposer grid`.
- `fixed_benchmark.py`, `pretrain_eval.py`, shard preparation, scoring formula,
  time budget, and benchmark truncation stay fixed during an experiment.
- `data/shards/{train,val}.bin` are the immutable tokenized surfaces for a run.
- `results.tsv` is the experiment ledger. Append to it; `--reset` is the only way to clear it.
- `papers.tsv` is the literature ledger. Never silently delete or rewrite it.
- `scores.tsv` is the model scoreboard (our runs + published references).
- `best.json` / `checkpoints/best/` are the last accepted candidate and weights.

## Loop

1. Optionally resume from `checkpoints/best`.
2. Run a baseline before changing anything.
3. Make one intelligible mutation, cited to a logged paper when the change
   comes from the literature. `--proposer grok` may rewrite `model.py`.
   `--proposer grid` only walks the candidate hyperparameters.
4. Train for the same wall-clock budget on FineWeb shards.
5. Measure held-out BPB and truncated MMLU/ARC/OpenBookQA accuracy.
6. Keep the mutation only when its score improves by at least 0.001.
7. Log every completed experiment and update paper status when applicable.
