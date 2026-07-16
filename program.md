# Autoresearch operating contract

The objective is to lower the fixed `score` reported by `fixed_benchmark.py`.
The score combines validation loss with small deployment-latency and parameter
costs. Lower is better. A candidate that cannot execute on the selected
deployment device is invalid.

## Files and boundaries

- `candidate.json` is the only automatically mutable research surface.
- `fixed_benchmark.py`, the dataset seed, scoring formula, time budget, and
  deployment requirement are fixed during an experiment.
- `results.tsv` is the experiment ledger. Never silently delete or rewrite a
  prior result.
- `best.json` is the last accepted candidate.

## Loop

1. Run a baseline before changing anything.
2. Make one small, intelligible mutation from the current best candidate.
3. Train for the same wall-clock budget and evaluate with the fixed validation
   set.
4. Time inference for the trained weights on the selected deployment device
   (NumPy CPU or PyTorch CUDA).
5. Keep the mutation only when its score improves by at least 0.001. This
   guards against timing and wall-clock-step noise. Otherwise restore the
   current best.
6. Log every completed experiment, including discarded experiments.
7. Prefer a simpler candidate when scores are effectively tied.

The built-in proposer searches optimizer and width settings. A coding agent may
extend the candidate representation or model family, but it must not weaken the
fixed metric, leak validation labels, change seeds per candidate, extend the
time budget, or replace GPU execution with CPU while claiming accelerator
success.

## Hardware interpretation

On an NVIDIA computer, PyTorch CUDA performs both the training experiment and
deployment timing. On machines without CUDA, NumPy CPU is used for both.
