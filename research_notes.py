"""One markdown note per tried idea, kept in research/."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research"


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    body: list[str] = []
    collecting = False
    for line in lines:
        if line.strip() == f"## {heading}":
            collecting = True
            continue
        if collecting and line.startswith("## "):
            break
        if collecting:
            body.append(line)
    return "\n".join(body).strip()


def note_path(paper_id: str) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", paper_id.strip()) or "note"
    return RESEARCH / f"{safe}.md"


def next_draft_path() -> Path:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in RESEARCH.glob("proposal_draft_*.md"):
        match = re.fullmatch(r"proposal_draft_(\d+)", path.stem)
        if match:
            numbers.append(int(match.group(1)))
    return RESEARCH / f"proposal_draft_{max(numbers, default=0) + 1}.md"


def existing_ids() -> set[str]:
    if not RESEARCH.exists():
        return set()
    return {path.stem for path in RESEARCH.glob("*.md")}


def write_note(
    path: Path,
    *,
    title: str,
    abstract: str,
    core_idea: str,
    math_basis: str,
    source: str,
    ident: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"# {title.strip() or ident}\n\n"
            f"- id: {ident}\n"
            f"- source: {source}\n"
            f"- status: pending\n\n"
            f"## Abstract\n\n{abstract.strip() or 'Not stated.'}\n\n"
            f"## Core idea\n\n{core_idea.strip() or 'Not stated.'}\n\n"
            f"## Mathematical basis\n\n{math_basis.strip() or 'Not stated.'}\n\n"
            f"## Decision\n\npending\n"
        ),
        encoding="utf-8",
    )
    return path


def set_decision(path: Path, decision: str, detail: str = "") -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^(- status:).*$", rf"\1 {decision}", text, count=1, flags=re.M)
    block = f"## Decision\n\n{decision}"
    if detail.strip():
        block += f"\n\n{detail.strip()}"
    block += "\n"
    if "## Decision" in text:
        text = re.sub(r"## Decision\s.*\Z", block, text, count=1, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def context_for_prompt(limit: int = 8000) -> str:
    """Titles and core ideas already tried, oldest ledger rows first."""
    chunks: list[str] = []
    from papers import latest_status_by_id

    noted = existing_ids()
    for paper_id, row in latest_status_by_id(ROOT / "papers.tsv").items():
        if paper_id in noted:
            continue
        title = " ".join((row.get("title") or "").split())
        idea = " ".join((row.get("idea") or "").split())[:180]
        if title or idea:
            chunks.append(f"- {paper_id}: {title}. {idea}")
    if RESEARCH.exists():
        for path in sorted(RESEARCH.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
            core = " ".join(_section(text, "Core idea").split())
            chunks.append(f"- {path.name}: {title}. {core}")
    blob = "\n".join(chunks)
    if len(blob) > limit:
        blob = blob[-limit:]
    return blob
