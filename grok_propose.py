"""Ask Grok for one architecture or method change to model.py.

Mirrors the Tesseract research loop: one full-file update, a one-line idea,
and a paper citation. The evaluator, score, and data shards stay fixed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
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


def chat(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 300.0,
    max_tokens: int = 16000,
) -> str:
    body = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "stream": True,
        "messages": messages,
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        XAI_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    text = ""
    last_error: Exception | None = None
    for attempt in range(3):
        content: list[str] = []
        reasoning: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    thought = delta.get("reasoning_content")
                    if isinstance(piece, str):
                        content.append(piece)
                    if isinstance(thought, str):
                        reasoning.append(thought)
            text = "".join(content).strip()
            if not text:
                text = "".join(reasoning).strip()
            if text:
                break
            last_error = RuntimeError("xAI returned an empty message")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:800]
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f"xAI HTTP {error.code}: {detail}") from error
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 2:
                raise RuntimeError(f"xAI request failed: {error}") from error
        time.sleep(2.0 * (attempt + 1))
    if not text:
        raise RuntimeError(f"xAI request failed: {last_error}")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "last_api_response.json").write_text(text[:2_000_000], encoding="utf-8")
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


def extract_diff(text: str) -> str | None:
    fences = re.findall(r"```(?:diff|udiff|patch)?\s*\n(.*?)```", text, flags=re.S | re.I)
    for block in fences:
        if block.lstrip().startswith("---") or "\n---" in block:
            return block.strip() + "\n"
    return None


def apply_unified_diff(original: str, diff: str) -> str:
    work = ROOT / "runs" / "patch_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # Own repo so apply does not prefix paths with the parent checkout.
    subprocess.run(["git", "init", "-q"], cwd=work, check=True, capture_output=True)
    target = work / "model.py"
    original_lf = original.replace("\r\n", "\n")
    last_error = ""
    for strip in (1, 0):
        target.write_text(original_lf, encoding="utf-8", newline="\n")
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", f"-p{strip}"],
            input=diff.replace("\r\n", "\n").encode("utf-8"),
            cwd=work,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and target.exists():
            patched = target.read_text(encoding="utf-8")
            shutil.rmtree(work, ignore_errors=True)
            return patched
        last_error = (completed.stderr or completed.stdout).decode("utf-8", errors="replace")[-500:]
    shutil.rmtree(work, ignore_errors=True)
    raise RuntimeError(last_error or "git apply failed")


def _parse_preamble(text: str) -> tuple[str, str, str, str]:
    idea = ""
    paper_id = ""
    title = ""
    summary = ""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if upper == "IDEA" or upper.startswith("IDEA:"):
            idea = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if not idea:
                for follow in lines[index + 1 :]:
                    follow = follow.strip()
                    if not follow:
                        continue
                    if follow.upper().startswith("PAPER") or follow.startswith("```"):
                        break
                    idea = follow
                    break
        elif upper == "PAPER" or upper.startswith("PAPER:"):
            body = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if not body:
                for follow in lines[index + 1 :]:
                    follow = follow.strip()
                    if follow and not follow.startswith("```"):
                        body = follow
                        break
            parts = [part.strip() for part in body.split("|")]
            if parts and parts[0]:
                match = re.search(r"\d{4}\.\d{4,5}", parts[0])
                paper_id = match.group(0) if match else ""
                title = parts[1] if len(parts) > 1 else parts[0]
                summary = parts[2] if len(parts) > 2 else idea
            break
        elif stripped.startswith("```"):
            break
    if not idea:
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("```") and stripped.upper() not in {"IDEA", "PAPER"}:
                idea = stripped[:180]
                break
    return idea[:180], paper_id, title[:120], (summary or idea)[:180]


def validate_source(source: str) -> None:
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in source]
    if missing:
        raise RuntimeError(f"proposal is missing {missing}")
    if "fixed_benchmark" in source or "overall_score" in source:
        raise RuntimeError("proposal must not rewrite the fixed score")


def choose_method_query(
    api_key: str,
    *,
    model: str,
    results_tail: str,
    papers_tail: str,
    avoid: str = "",
) -> str:
    """Ask for a search query. The arXiv id comes from search, not from the model."""
    avoid_note = f"\nDo not search for: {avoid}\n" if avoid else ""
    reply = chat(
        api_key,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You pick a literature search for one method that could lower a small "
                    "language model's pretrain score (bits-per-byte plus benchmark error). "
                    "Methods include attention, feed-forward, normalization, position encoding, "
                    "and initialization from models such as Kimi, Qwen, Mistral, DeepSeek, or Gemma. "
                    "Do not invent an arXiv id. Reply with one line: QUERY: <search words>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Recent results:\n{results_tail}\n\n"
                    f"Papers already logged:\n{papers_tail}\n"
                    f"{avoid_note}"
                    "QUERY:"
                ),
            },
        ],
        max_tokens=128,
    )
    match = re.search(r"QUERY:\s*(.+)", reply)
    query = (match.group(1) if match else reply).strip().splitlines()[0]
    query = re.sub(r"\d{4}\.\d{4,5}", "", query).strip(" :|-")
    if not query:
        raise RuntimeError("Grok did not return a method search query")
    return query[:180]


def find_method_paper(
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    results_tail: str = "",
    papers_tail: str = "",
    known: set[str] | None = None,
) -> dict[str, str]:
    """Search papers for a method. The returned arXiv id is from the search hit."""
    from papers import search_papers

    seen = set(known or ())
    avoid = ""
    last_query = ""
    for _ in range(3):
        last_query = choose_method_query(
            api_key,
            model=model,
            results_tail=results_tail,
            papers_tail=papers_tail,
            avoid=avoid,
        )
        hits = search_papers(last_query, limit=15)
        for hit in hits:
            if hit["arxiv_id"] in seen:
                continue
            return {
                "arxiv_id": hit["arxiv_id"],
                "title": hit["title"],
                "summary": str(hit.get("summary") or ""),
                "query": last_query,
                "url": hit["url"],
            }
        avoid = last_query
    raise RuntimeError(f"no unseen paper for method search {last_query!r}")


def _proposal_source(reply: str, model_src: str) -> str | None:
    diff = extract_diff(reply)
    if diff is not None:
        try:
            patched = apply_unified_diff(model_src, diff)
        except RuntimeError:
            patched = ""
        if patched and all(snippet in patched for snippet in REQUIRED_SNIPPETS):
            return patched if patched.endswith("\n") else patched + "\n"
    return extract_model_source(reply)


def propose_architecture(
    api_key: str,
    model_src: str,
    paper: dict[str, str],
    *,
    model: str = DEFAULT_MODEL,
    results_tail: str = "",
    candidate_json: str = "",
    best_score: float | None = None,
) -> ArchitectureProposal:
    paper_id = paper["arxiv_id"]
    system = (
        "You edit model.py for a small decoder-only language model trained on FineWeb. "
        "The training loop, score, tokenizer, and data shards are fixed. "
        "Implement one method from the paper that was already retrieved by search. "
        "Do not swap in a different paper or invent an arXiv id. "
        "Do not only retune learning rate, width, depth, or weight decay — those are overridden "
        "by candidate.json. New ModelConfig fields are allowed when they have defaults and the "
        "modules read them from config. Keep the public classes ModelConfig, TransformerLM, "
        "and CausalLMOutput, and the methods forward, configure_optimizer, get_num_params, "
        "and gradient_checkpointing_enable. Vocab size stays 50304. The model must train "
        "with micro-batches on a 6GB GPU. One change only. "
        "Keep every existing parameter name and shape so the current checkpoint still loads. "
        "Change the computation, not tensor sizes. "
        "Reply with one header line and a unified diff. Do not return the whole file.\n"
        "IDEA: one sentence of what this trial changes\n"
        "```diff\n--- a/model.py\n+++ b/model.py\n<unified diff>\n```"
    )
    abstract = (paper.get("summary") or "")[:1200]
    user = (
        f"Paper from search (id is fixed): {paper_id}\n"
        f"Title: {paper.get('title', '')}\n"
        f"Abstract: {abstract}\n\n"
        f"Best score so far (lower is better): {best_score}\n"
        f"Current candidate.json (size and optimizer; do not spend this trial on these knobs):\n"
        f"{candidate_json}\n\n"
        f"Recent results.tsv:\n{results_tail}\n\n"
        f"Current model.py:\n```python\n{model_src}\n```"
    )
    reply = chat(
        api_key,
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=4000,
    )
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "last_proposal.md").write_text(reply, encoding="utf-8")
    source = _proposal_source(reply, model_src)
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
        source = _proposal_source(reply, model_src)
    if source is None:
        raise RuntimeError("Grok did not return a usable model.py")
    validate_source(source)
    idea, _ignored_id, _ignored_title, summary = _parse_preamble(reply)
    return ArchitectureProposal(
        source=source,
        idea=idea or "architecture change",
        paper_id=paper["arxiv_id"],
        paper_title=paper.get("title") or "",
        summary=summary or idea or "architecture change",
    )
