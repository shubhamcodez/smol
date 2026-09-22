# AGI model

`model.py` defines a 203,821,824-parameter decoder-only Transformer targeted at
training and inference on an RTX 3070 with 6GB VRAM. Its maximum context length
is 4,096 tokens. The CUDA path cannot be hardware-verified on the Snapdragon
development machine, so run the included smoke test on the 3070 before a long
training job.

## Default architecture

- 24 decoder layers
- 768 hidden width
- 12 query heads and 4 key/value heads
- 2,304-wide SwiGLU feed-forward layers
- 50,304-token tied vocabulary
- RoPE, RMSNorm, GQA, PyTorch SDPA/Flash Attention, and KV caching

PyTorch 2.5 or newer is required because the attention path uses native GQA
without expanding key/value heads in VRAM.

## What is distinctive about this design

The individual Transformer components are established techniques. What is
distinctive here is how they are combined around one concrete constraint: full
parameter training with a 4,096-token window on a 6GB consumer GPU.

| Design choice | What this implementation does | Why it matters |
|---|---|---|
| Capacity chosen from the hardware budget | Uses exactly 203,821,824 parameters rather than scaling an arbitrary standard model | Leaves VRAM for 4K activations, optimizer state, and CUDA workspace |
| Native grouped-query attention | Uses 12 query heads but only 4 key/value heads through PyTorch `enable_gqa` | Reduces KV projections and cache memory without explicitly copying KV heads on the normal CUDA path |
| Two-level activation recomputation | Checkpoints every decoder block and separately checkpoints each vocabulary-loss chunk | Avoids retaining both the full block activations and a `[batch, 4096, 50304]` logits graph |
| Optional training logits | `return_logits=False` calculates the loss without recreating the complete output tensor | Saves hundreds of megabytes at full context while inference still returns logits normally |
| Compact generation cache | Stores only the four KV heads and computes only the newest token's logits during cached generation | Keeps autoregressive inference memory proportional to the compact KV representation |
| Tied vocabulary matrix | Shares token embeddings with the language-model output head | Removes a second 38,633,472-parameter vocabulary matrix |
| Stable low-precision numerics | RMSNorm accumulates variance in FP32, residual projections use depth-scaled initialization, and training uses FP16 with gradient scaling | Targets stable optimization without keeping activation tensors in FP32 |
| Optimizer-aware parameter grouping | Applies weight decay to matrices, excludes vectors/norm parameters, and selects fused AdamW on CUDA when available | Avoids inappropriate norm decay and reduces optimizer overhead |
| Exact analytical sizing | `default_parameter_count()` computes the parameter count without first allocating the model | Allows architecture search to reject oversized candidates before consuming VRAM |

The model remains bias-free and dropout-free by default. RoPE supplies positions
without a learned 4,096-row position table, pre-normalized RMSNorm protects the
residual stream, and SwiGLU provides the gated feed-forward path.

## Autoresearch design

The research harness follows a constrained experimental design rather than
allowing an agent to change everything at once:

- the evaluator, validation shards, time budget, seeds, and score are immutable;
- `candidate.json` is the small automatically mutable surface (width, depth, LR, …);
- every run starts with a baseline and changes one intelligible variable;
- improved candidates are kept, regressions and crashes are recorded and
  discarded, and `results.tsv` remains the experiment ledger;
- a minimum improvement threshold prevents ordinary timing noise from becoming
  a false discovery;
- fixed wall-clock trials reward changes that improve quality as well as changes
  that process more useful tokens in the same time.

### Pretrain score

```
score = bpb + 2.0 * (1 - mean_benchmark_acc) + 1e-9 * parameter_count
```

Lower is better. `bpb` is held-out FineWeb bits-per-byte; `mean_benchmark_acc`
averages truncated MMLU, ARC-Challenge, and OpenBookQA (log-likelihood MCQ).

### Literature + checkpoints

Ideas come from top-venue papers, including architecture and methods from
models such as Kimi, Qwen, and Mistral. `--proposer grok` asks Grok for one
change to `model.py`, trains it under the fixed score, and keeps or restores
the file. `--proposer grid` only walks hyperparameters in `candidate.json`.
Log every referral in `papers.tsv` so the same paper is not re-tried blindly.
Accepted weights land in `checkpoints/best/` for later resume.

```powershell
python .\papers.py next --venue iclr --query "pretraining data filtering"
python .\papers.py search "neurips fineweb" --log-new
python .\papers.py fetch 2406.17557 --markdown
python .\papers.py log 2406.17557 --venue neurips --status read --idea "edu quality filter"
python .\loop.py --proposer grok --iterations 4 --budget-seconds 30 --training-backend cuda
python .\loop.py --proposer grid --resume --iterations 4 --budget-seconds 30 --training-backend cuda
python .\papers.py list
python .\scores.py seed
python .\scores.py compare
```

`scores.tsv` stores our BPB/score/truncated-bench numbers alongside published
reference figures (Qwen2.5, Llama-3, Mistral, …). Protocols differ — compare
within `protocol`, not across.

See `program.md` for the full operating contract.

### Setup and run

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python .\prepare_data.py --train-tokens 50000000 --val-tokens 1000000
python .\nvidia_smoke.py
python .\loop.py --reset --iterations 4 --budget-seconds 30 --training-backend cuda
python .\audit.py
```

`prepare_data.py` writes `data/shards/train.bin` and `val.bin`. Benchmarks live
under `benchmarks/`.

For full-length training, use FP16 automatic mixed precision, micro-batch size
1, gradient accumulation, and both checkpointing mechanisms:

```python
import torch

from model import ModelConfig, TransformerLM

model = TransformerLM(ModelConfig()).cuda()
model.gradient_checkpointing_enable()
optimizer = model.configure_optimizer(0.1, 3e-4, device_type="cuda")
scaler = torch.amp.GradScaler("cuda")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(
        input_ids,
        targets=targets,
        return_logits=False,
        loss_chunk_size=256,
    )
scaler.scale(output.loss).backward()
scaler.step(optimizer)
scaler.update()
```

`return_logits=False` avoids recreating the complete logits output after the
chunked loss has been calculated. Sequence lengths below 4,096 reduce memory
and train faster without changing the model weights.

On the NVIDIA computer, verify checkpointed training and cached inference with:

```powershell
python .\nvidia_smoke.py
python .\nvidia_smoke.py --full --sequence-length 4096
```

The first command is a quick CUDA-path check. The second performs one complete
optimizer step with the default model and reports peak allocated and reserved
VRAM. Run the full check before beginning a long training job.
