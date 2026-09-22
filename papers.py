"""Append-only literature ledger for conference-paper driven research.

Status values:
  seen      - discovered in search, not yet read
  read      - paper markdown/abstract reviewed
  applied   - an idea from the paper was tried in a run
  skipped   - intentionally not pursued (duplicate idea, out of scope, …)
  rejected  - tried or considered; do not reopen without new evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "papers.tsv"
VENUES = ("iclr", "neurips", "nips", "emnlp", "aaai", "acl", "icml", "cvpr", "naacl", "colm")

COLUMNS = [
    "timestamp_utc",
    "arxiv_id",
    "title",
    "venue",
    "year",
    "status",
    "idea",
    "run_id",
    "notes",
    "url",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_arxiv_id(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^https?://(www\.)?(arxiv\.org/(abs|pdf)/|huggingface\.co/papers/)", "", text)
    text = text.removesuffix(".pdf").removesuffix(".md")
    text = text.split("?")[0].split("#")[0]
    # Drop version suffix for ledger identity so v1/v2 collapse.
    match = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", text)
    if match:
        return match.group(1)
    return text


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def known_ids(path: Path) -> set[str]:
    return {row["arxiv_id"] for row in load_rows(path) if row.get("arxiv_id")}


def append_row(path: Path, row: dict[str, str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in COLUMNS})


def http_get_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smol-autoresearch/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smol-autoresearch/1.0", "Accept": "text/markdown, text/plain"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_paper_meta(arxiv_id: str) -> dict[str, Any]:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    payload = http_get_json(f"https://huggingface.co/api/papers/{urllib.parse.quote(arxiv_id)}")
    paper = payload.get("paper", payload)
    title = paper.get("title") or ""
    summary = paper.get("summary") or paper.get("ai_summary") or ""
    published = str(paper.get("publishedAt") or paper.get("published_at") or "")
    year = published[:4] if published else ""
    venue = ""
    for key in ("conference", "venue", "publishedIn"):
        if paper.get(key):
            venue = str(paper[key])
            break
    return {
        "arxiv_id": arxiv_id,
        "title": title.replace("\t", " ").replace("\n", " ").strip(),
        "venue": venue.replace("\t", " ").strip(),
        "year": year,
        "summary": summary,
        "url": f"https://huggingface.co/papers/{arxiv_id}",
    }


def search_papers(query: str, limit: int = 20) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"q": query, "limit": str(limit)})
    payload = http_get_json(f"https://huggingface.co/api/papers/search?{params}")
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("papers") or payload.get("results") or payload.get("data") or []
    else:
        items = []
    results = []
    for item in items:
        paper = item.get("paper", item) if isinstance(item, dict) else {}
        arxiv_id = normalize_arxiv_id(
            str(paper.get("id") or paper.get("arxiv_id") or item.get("id") or "")
        )
        if not arxiv_id:
            continue
        title = str(paper.get("title") or "").replace("\t", " ").replace("\n", " ").strip()
        published = str(paper.get("publishedAt") or paper.get("published_at") or "")
        results.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "year": published[:4],
                "url": f"https://huggingface.co/papers/{arxiv_id}",
                "summary": paper.get("summary") or paper.get("ai_summary") or "",
            }
        )
    return results


def cmd_search(args: argparse.Namespace) -> None:
    known = known_ids(args.log)
    query = args.query
    if args.venue:
        query = f"{args.venue} {query}".strip()
    hits = search_papers(query, limit=args.limit)
    fresh = 0
    for hit in hits:
        seen = hit["arxiv_id"] in known
        marker = "OLD" if seen else "NEW"
        if not seen:
            fresh += 1
        print(f"[{marker}] {hit['arxiv_id']}\t{hit['year']}\t{hit['title']}")
        if args.log_new and not seen:
            append_row(
                args.log,
                {
                    "timestamp_utc": utc_now(),
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit["title"],
                    "venue": args.venue or "",
                    "year": hit["year"],
                    "status": "seen",
                    "idea": "",
                    "run_id": "",
                    "notes": f"search:{query}",
                    "url": hit["url"],
                },
            )
            known.add(hit["arxiv_id"])
    print(f"SEARCH_HITS={len(hits)} NEW={fresh} LOGGED={args.log_new}")


def cmd_log(args: argparse.Namespace) -> None:
    arxiv_id = normalize_arxiv_id(args.arxiv_id)
    if arxiv_id in known_ids(args.log) and not args.force:
        print(f"ALREADY_LOGGED {arxiv_id}")
        return
    meta = fetch_paper_meta(arxiv_id)
    append_row(
        args.log,
        {
            "timestamp_utc": utc_now(),
            "arxiv_id": meta["arxiv_id"],
            "title": meta["title"],
            "venue": args.venue or meta["venue"],
            "year": meta["year"],
            "status": args.status,
            "idea": args.idea or "",
            "run_id": args.run_id or "",
            "notes": args.notes or "",
            "url": meta["url"],
        },
    )
    print(f"LOGGED {meta['arxiv_id']}\t{meta['title']}")


def cmd_list(args: argparse.Namespace) -> None:
    rows = load_rows(args.log)
    if args.status:
        rows = [row for row in rows if row.get("status") == args.status]
    for row in rows:
        print(
            f"{row.get('status',''):8}\t{row.get('arxiv_id','')}\t"
            f"{row.get('venue','')}\t{row.get('title','')}"
        )
    print(f"COUNT={len(rows)}")


def cmd_mark(args: argparse.Namespace) -> None:
    arxiv_id = normalize_arxiv_id(args.arxiv_id)
    rows = load_rows(args.log)
    if not any(row.get("arxiv_id") == arxiv_id for row in rows):
        raise SystemExit(f"paper {arxiv_id} is not in {args.log}; run papers.py log first")
    # Append a new status row (append-only ledger; latest wins for readers).
    latest = next(row for row in reversed(rows) if row.get("arxiv_id") == arxiv_id)
    append_row(
        args.log,
        {
            "timestamp_utc": utc_now(),
            "arxiv_id": arxiv_id,
            "title": latest.get("title", ""),
            "venue": latest.get("venue", ""),
            "year": latest.get("year", ""),
            "status": args.status,
            "idea": args.idea if args.idea is not None else latest.get("idea", ""),
            "run_id": args.run_id if args.run_id is not None else latest.get("run_id", ""),
            "notes": args.notes if args.notes is not None else latest.get("notes", ""),
            "url": latest.get("url", f"https://huggingface.co/papers/{arxiv_id}"),
        },
    )
    print(f"MARKED {arxiv_id} -> {args.status}")


def cmd_fetch(args: argparse.Namespace) -> None:
    arxiv_id = normalize_arxiv_id(args.arxiv_id)
    if arxiv_id in known_ids(args.log) and not args.allow_reread:
        print(f"ALREADY_IN_LOG {arxiv_id} (pass --allow-reread to fetch anyway)")
    meta = fetch_paper_meta(arxiv_id)
    print(json.dumps({k: meta[k] for k in ("arxiv_id", "title", "year", "venue", "url")}, indent=2))
    print("\n--- ABSTRACT ---\n")
    print(meta.get("summary") or "(no summary)")
    if args.markdown:
        print("\n--- MARKDOWN (truncated) ---\n")
        text = http_get_text(f"https://huggingface.co/papers/{arxiv_id}.md")
        print(text[: args.markdown_chars])


def cmd_next(args: argparse.Namespace) -> None:
    """Suggest NEW search hits not already in the ledger."""
    known = known_ids(args.log)
    query = args.query or "language model pretraining efficiency"
    if args.venue:
        query = f"{args.venue} {query}"
    hits = [hit for hit in search_papers(query, limit=args.limit) if hit["arxiv_id"] not in known]
    for hit in hits[: args.show]:
        print(f"{hit['arxiv_id']}\t{hit['year']}\t{hit['title']}\t{hit['url']}")
    print(f"UNSEEN={len(hits)}")


def latest_status_by_id(path: Path) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in load_rows(path):
        if row.get("arxiv_id"):
            latest[row["arxiv_id"]] = row
    return latest


def _one_line(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def attempt_summaries(paper_ids: list[str], path: Path = DEFAULT_LOG) -> list[str]:
    """One line per paper: id, short title, venue, and the idea being tried."""
    latest = latest_status_by_id(path)
    lines: list[str] = []
    for raw_id in paper_ids:
        paper_id = normalize_arxiv_id(raw_id)
        row = latest.get(paper_id)
        if row is None:
            lines.append(f"{paper_id}: not in papers.tsv")
            continue
        title = _one_line(row.get("title", ""), 72)
        title = title.split(":")[0].strip() or title
        idea = _one_line(row.get("idea", "") or "idea not logged")
        where = ", ".join(part for part in (row.get("venue", ""), row.get("year", "")) if part)
        where_bit = f" ({where})" if where else ""
        lines.append(f"{paper_id}{where_bit} {title}: {idea}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Conference paper research ledger")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search HF papers; optionally log NEW hits")
    search.add_argument("query")
    search.add_argument("--venue", choices=VENUES, default=None)
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--log-new", action="store_true")
    search.set_defaults(func=cmd_search)

    log = sub.add_parser("log", help="append one paper to the ledger")
    log.add_argument("arxiv_id")
    log.add_argument("--venue", default="")
    log.add_argument("--status", default="seen", choices=["seen", "read", "applied", "skipped", "rejected"])
    log.add_argument("--idea", default="")
    log.add_argument("--run-id", default="")
    log.add_argument("--notes", default="")
    log.add_argument("--force", action="store_true")
    log.set_defaults(func=cmd_log)

    listing = sub.add_parser("list", help="list logged papers")
    listing.add_argument("--status", default=None)
    listing.set_defaults(func=cmd_list)

    mark = sub.add_parser("mark", help="append a status update for a logged paper")
    mark.add_argument("arxiv_id")
    mark.add_argument("--status", required=True, choices=["seen", "read", "applied", "skipped", "rejected"])
    mark.add_argument("--idea", default=None)
    mark.add_argument("--run-id", default=None)
    mark.add_argument("--notes", default=None)
    mark.set_defaults(func=cmd_mark)

    fetch = sub.add_parser("fetch", help="fetch metadata/abstract; refuse silent rereads")
    fetch.add_argument("arxiv_id")
    fetch.add_argument("--markdown", action="store_true")
    fetch.add_argument("--markdown-chars", type=int, default=4000)
    fetch.add_argument("--allow-reread", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    nxt = sub.add_parser("next", help="list unseen search hits for a venue/topic")
    nxt.add_argument("--query", default="language model pretraining")
    nxt.add_argument("--venue", choices=VENUES, default="iclr")
    nxt.add_argument("--limit", type=int, default=30)
    nxt.add_argument("--show", type=int, default=10)
    nxt.set_defaults(func=cmd_next)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
