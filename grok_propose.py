"""Ask Grok for one architecture or method change to model.py.

Mirrors the Tesseract research loop: one full-file update, a one-line idea,
and a paper citation. The evaluator, score, and data shards stay fixed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
XAI_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.6"

REQUIRED_SNIPPETS = (
    "class ModelConfig",
    "class TransformerLM",
    "def forward",
    "def configure_optimizer",
    "def get_num_params",
    "class CausalLMOutput",
)


@dataclass(frozen=True)
class ArchitectureProposal:
    source: str
    idea: str
    paper_id: str
    paper_title: str
    summary: str

    @property
    def paper_line(self) -> str:
        if not self.paper_id and not self.paper_title:
            return self.summary
        title = self.paper_title or self.paper_id
        idea = self.summary or self.idea
        ident = self.paper_id or "uncited"
        return f"{ident} {title}: {idea}"


def load_api_key(path: Path = ENV_PATH) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "XAI_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def _message_text(message: dict) -> str:
    def as_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            bits: list[str] = []
            for item in value:
                if isinstance(item, str):
                    bits.append(item)
                elif isinstance(item, dict):
                    bits.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(bits)
        return str(value)

    content = as_text(message.get("content"))
    reasoning = as_text(message.get("reasoning_content") or message.get("reasoning"))
    parts: list[str] = []
    if content.strip():
        parts.append(content)
    if reasoning.strip() and reasoning.strip() not in content:
        parts.append(reasoning)
    return "\n".join(parts)


def chat(api_key: str, model: str, messages: list[dict[str, str]], timeout: float = 300.0) -> str:
    payload = json.dumps(
        {"model": model, "temperature": 0.3, "max_tokens": 16000, "messages": messages}
    ).encode("utf-8")
    request = urllib.request.Request(
        XAI_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"xAI HTTP {error.code}: {detail}") from error
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "last_api_response.json").write_text(json.dumps(data)[:2_000_000], encoding="utf-8")
    text = _message_text(data["choices"][0]["message"])
    if not text.strip():
        raise RuntimeError("xAI returned an empty message")
    return text


def extract_model_source(text: str) -> str | None:
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    candidates = fences or []
    if not candidates:
        open_fence = re.search(r"```(?:python|py)?\s*\n(.*)\Z", text, flags=re.S | re.I)
        if open_fence:
            candidates.append(open_fence.group(1))
    for block in candidates:
        source = block.strip() + "\n"
        if all(snippet in source for snippet in REQUIRED_SNIPPETS):
            return source
    return None


def _parse_preamble(text: str) -> tuple[str, str, str, str]:
    idea = ""
    paper_id = ""
    title = ""
    summary = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("IDEA:"):
            idea = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("PAPER:"):
            body = stripped.split(":", 1)[1].strip()
            parts = [part.strip() for part in body.split("|")]
            if parts:
                match = re.search(r"\d{4}\.\d{4,5}", parts[0])
                paper_id = match.group(0) if match else ""
                title = parts[1] if len(parts) > 1 else parts[0]
                summary = parts[2] if len(parts) > 2 else idea
            break
        if stripped.startswith("```"):
            break
    if not idea:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("```") and not stripped.upper().startswith("PAPER:"):
                idea = stripped[:180]
                break
    return idea[:180], paper_id, title[:120], (summary or idea)[:180]


def validate_source(source: str) -> None:
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in source]
    if missing:
        raise RuntimeError(f"proposal is missing {missing}")
    if "fixed_benchmark" in source or "overall_score" in source:
        raise RuntimeError("proposal must not rewrite the fixed score")


def propose_architecture(
    api_key: str,
    model_src: str,
    *,
    model: str = DEFAULT_MODEL,
    results_tail: str = "",
    papers_tail: str = "",
    candidate_json: str = "",
    best_score: float | None = None,
) -> ArchitectureProposal:
    system = (
        "You edit model.py for a small decoder-only language model trained on FineWeb. "
        "The training loop, score, tokenizer, and data shards are fixed. "
        "You may change architecture and methods: attention, feed-forward, norms, "
        "positions, initialization, or a technique from a real paper "
        "(Kimi, Qwen, Mistral, DeepSeek, Gemma, and similar). "
        "Do not only retune learning rate, width, depth, or weight decay — those are overridden "
        "by candidate.json. New ModelConfig fields are allowed when they have defaults and the "
        "modules read them from config. Keep the public classes ModelConfig, TransformerLM, "
        "and CausalLMOutput, and the methods forward, configure_optimizer, get_num_params, "
        "and gradient_checkpointing_enable. Vocab size stays 50304. The model must train "
        "with micro-batches on a 6GB GPU. One change only. "
        "Reply with exactly two header lines and then the complete file:\n"
        "IDEA: one sentence of what this trial changes\n"
        "PAPER: arxiv_id | short title | one-line summary of the idea being tried\n"
        "```python\n<full model.py>\n```"
    )
    user = (
        f"Best score so far (lower is better): {best_score}\n"
        f"Current candidate.json (size and optimizer; do not spend this trial on these knobs):\n"
        f"{candidate_json}\n\n"
        f"Recent results.tsv:\n{results_tail}\n\n"
        f"Papers already tried (do not repeat a rejected idea):\n{papers_tail}\n\n"
        f"Current model.py:\n```python\n{model_src}\n```"
    )
    reply = chat(
        api_key,
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "last_proposal.md").write_text(reply, encoding="utf-8")
    source = extract_model_source(reply)
    if source is None:
        reply = chat(
            api_key,
            model,
            [
                {
                    "role": "system",
                    "content": "Output only IDEA, PAPER, and one python fence containing the full model.py.",
                },
                {
                    "role": "user",
                    "content": (
                        "The previous reply had no usable model.py. Return the full file with "
                        "ModelConfig, TransformerLM, CausalLMOutput, forward, configure_optimizer, "
                        "and get_num_params preserved. One architecture change from a cited paper.\n\n"
                        f"```python\n{model_src}\n```"
                    ),
                },
            ],
        )
        (runs / "last_proposal.md").write_text(reply, encoding="utf-8")
        source = extract_model_source(reply)
    if source is None:
        raise RuntimeError("Grok did not return a usable model.py")
    validate_source(source)
    idea, paper_id, title, summary = _parse_preamble(reply)
    return ArchitectureProposal(
        source=source,
        idea=idea or "architecture change",
        paper_id=paper_id,
        paper_title=title,
        summary=summary or idea or "architecture change",
    )
