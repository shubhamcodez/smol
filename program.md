# Autoresearch operating contract

The objective is to lower the fixed `score` reported by `fixed_benchmark.py`.

```
score = bpb + 2.0 * (1 - mean_benchmark_acc) + 1e-9 * parameter_count
```

`bpb` is held-out FineWeb bits-per-byte. `mean_benchmark_acc` averages truncated
MMLU, ARC-Challenge, and OpenBookQA multiple-choice accuracy scored by
log-likelihood. Lower is better.

## Files and boundaries

- `candidate.json` is the only automatically mutable research surface.
- `fixed_benchmark.py`, `pretrain_eval.py`, shard preparation, scoring formula,
  time budget, and benchmark truncation are fixed during an experiment.
- `data/shards/{train,val}.bin` are the immutable tokenized training/validation
  surfaces for a run (build once with `prepare_data.py`).
- `results.tsv` is the experiment ledger. Never silently delete or rewrite a
  prior result.
- `best.json` is the last accepted candidate.

## Loop

1. Run a baseline before changing anything.
2. Make one small, intelligible mutation from the current best candidate.
3. Train for the same wall-clock budget on FineWeb shards.
4. Measure held-out BPB and truncated MMLU/ARC/OpenBookQA accuracy.
5. Keep the mutation only when its score improves by at least 0.001.
6. Log every completed experiment, including discarded experiments.
7. Prefer a simpler candidate when scores are effectively tied (parameter term).

The built-in proposer searches learning rate, width, depth, weight decay, batch
size, sequence length, and Adam beta2. A coding agent may extend the candidate
representation, but it must not weaken the fixed metric, leak validation tokens
into training, change seeds per candidate, or extend the time budget silently.
