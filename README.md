# AGI model

Trying to make world's smallest viable model - the goal is to apply federated learning to train the model on smallest compute possible and compile to superintelligence.

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
python .\autoresearch\nvidia_smoke.py
python .\autoresearch\nvidia_smoke.py --full --sequence-length 4096
```

The first command is a quick CUDA-path check. The second performs one complete
optimizer step with the default model and reports peak allocated and reserved
VRAM. Run the full check before beginning a long training job.
