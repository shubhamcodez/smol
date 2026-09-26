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
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "papers.tsv"
VENUES = ("iclr", "neurips", "nips", "emnlp", "aaai", "acl", "icml", "cvpr", "naacl", "colm")
# Semantic Scholar venue names. arXiv is searched separately.
S2_VENUE = {
    "iclr": "ICLR",
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "emnlp": "EMNLP",
    "aaai": "AAAI",
    "acl": "ACL",
    "icml": "ICML",
    "cvpr": "CVPR",
    "naacl": "NAACL",
    "colm": "COLM",
}
# OpenReview group ids for venues hosted there. The rest go through Semantic Scholar.
OPENREVIEW_GROUP = {
    "iclr": "ICLR.cc",
    "neurips": "NeurIPS.cc",
    "nips": "NeurIPS.cc",
    "icml": "ICML.cc",
    "colm": "COLM.cc",
}

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


def _read_url(url: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smol-autoresearch/1.0 (local research)", "Accept": accept},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code != 429:
            raise
        time.sleep(3)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()


def http_get_json(url: str) -> Any:
    return json.loads(_read_url(url, "application/json").decode("utf-8"))


def http_get_text(url: str) -> str:
    return _read_url(url, "text/html, text/markdown, text/plain, */*").decode("utf-8", errors="replace")


def paper_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{normalize_arxiv_id(arxiv_id)}"


def _query_words(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,}", query)[:8]


def _short_venue(value: str) -> str:
    lowered = value.lower()
    if "corr" in lowered:
        return "arxiv"
    if "nips" in lowered and "neurips" not in lowered:
        return "neurips"
    for name in ("iclr", "neurips", "icml", "emnlp", "aaai", "naacl", "cvpr", "colm", "acl"):
        if name in lowered:
            return name
    return " ".join(value.split())


def _clean(value: object) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def _fetch_hf_meta(arxiv_id: str) -> dict[str, Any]:
    payload = http_get_json(f"https://huggingface.co/api/papers/{urllib.parse.quote(arxiv_id)}")
    paper = payload.get("paper", payload)
    title = _clean(paper.get("title"))
    summary = _clean(paper.get("summary") or paper.get("ai_summary"))
    published = str(paper.get("publishedAt") or paper.get("published_at") or "")
    venue = ""
    for key in ("conference", "venue", "publishedIn"):
        if paper.get(key):
            venue = _short_venue(str(paper[key]))
            break
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "venue": venue,
        "year": published[:4],
        "summary": summary,
        "url": paper_url(arxiv_id),
    }


def _meta_content(page: str, name: str) -> str:
    match = re.search(rf'<meta name="{re.escape(name)}" content="(.*?)"', page)
    return html.unescape(match.group(1)) if match else ""


def _fetch_arxiv_meta(arxiv_id: str) -> dict[str, Any]:
    page = http_get_text(f"https://arxiv.org/abs/{arxiv_id}")
    title = _meta_content(page, "citation_title")
    if not title:
        raise LookupError(f"arXiv has no record for {arxiv_id}")
    date = _meta_content(page, "citation_date")
    abstract_match = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', page, re.S)
    comments_match = re.search(r'class="tablecell comments[^"]*">(.*?)</td>', page, re.S)
    return {
        "arxiv_id": arxiv_id,
        "title": _clean(title),
        "venue": _short_venue(_clean(comments_match.group(1)) if comments_match else ""),
        "year": date[:4],
        "summary": re.sub(
            r"^Abstract:\s*",
            "",
            _clean(abstract_match.group(1) if abstract_match else ""),
        ),
        "url": paper_url(arxiv_id),
    }


def fetch_paper_meta(arxiv_id: str) -> dict[str, Any]:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    try:
        meta = _fetch_hf_meta(arxiv_id)
        if meta["title"]:
            return meta
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        pass
    return _fetch_arxiv_meta(arxiv_id)


def _hit(
    arxiv_id: str,
    title: str,
    year: str,
    summary: str,
    venue: str = "",
) -> dict[str, Any] | None:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    if not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
        return None
    return {
        "arxiv_id": arxiv_id,
        "title": _clean(title),
        "year": year[:4],
        "venue": _short_venue(venue) if venue else "",
        "url": paper_url(arxiv_id),
        "summary": _clean(summary),
    }


def _search_arxiv(query: str, limit: int, venue: str | None = None) -> list[dict[str, Any]]:
    words = _query_words(query)
    if venue:
        words.append(S2_VENUE.get(venue, venue))
    params = urllib.parse.urlencode(
        {
            "query": " ".join(words) or "transformer",
            "searchtype": "all",
            "source": "header",
            "start": "0",
            "size": str(min(max(limit, 1), 50)),
        }
    )
    page = http_get_text(f"https://arxiv.org/search/?{params}")
    results = []
    for block in page.split('class="arxiv-result"')[1:]:
        id_match = re.search(r"arxiv.org/abs/(\d{4}\.\d{4,5})", block)
        title_match = re.search(r'class="title is-5 mathjax">\s*(.*?)\s*</p>', block, re.S)
        abstract_match = re.search(r'class="abstract-short[^"]*"[^>]*>\s*(.*?)\s*</span>', block, re.S)
        date_match = re.search(r"originally announced</span>\s*[A-Za-z]+\s+(\d{4})", block)
        comments_match = re.search(r'class="comments[^"]*"[^>]*>(.*?)</p>', block, re.S)
        if not id_match or not title_match:
            continue
        hit = _hit(
            id_match.group(1),
            title_match.group(1),
            date_match.group(1) if date_match else "",
            abstract_match.group(1) if abstract_match else "",
            _clean(comments_match.group(1)) if comments_match else "",
        )
        if hit:
            results.append(hit)
        if len(results) >= limit:
            break
    return results


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(text).lower())


def _arxiv_id_for_title(title: str) -> str:
    """Resolve a conference paper to an arXiv id by an exact title search."""
    params = urllib.parse.urlencode(
        {
            "query": title,
            "searchtype": "title",
            "source": "header",
            "start": "0",
            "size": "1",
        }
    )
    page = http_get_text(f"https://arxiv.org/search/?{params}")
    blocks = page.split('class="arxiv-result"')
    if len(blocks) < 2:
        return ""
    id_match = re.search(r"arxiv.org/abs/(\d{4}\.\d{4,5})", blocks[1])
    title_match = re.search(r'class="title is-5 mathjax">\s*(.*?)\s*</p>', blocks[1], re.S)
    if not id_match or not title_match:
        return ""
    found = _norm_title(title_match.group(1))
    wanted = _norm_title(title)
    if not found or not wanted:
        return ""
    if found == wanted or found.startswith(wanted[:48]) or wanted.startswith(found[:48]):
        return id_match.group(1)
    return ""


def _search_openreview(query: str, limit: int, venue: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, str] = {
        "term": query,
        "limit": str(min(max(limit * 3, limit), 50)),
        "source": "forum",
    }
    group = OPENREVIEW_GROUP.get(venue or "")
    if group:
        params["group"] = group
    elif venue:
        params["term"] = f"{S2_VENUE.get(venue, venue)} {query}"
    payload = http_get_json("https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(params))
    results = []
    lookups = 0
    for note in payload.get("notes") or []:
        content = note.get("content") or {}
        blob = json.dumps(content)
        id_match = re.search(r"arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", blob) or re.search(
            r"abs-(\d{4}\.\d{4,5})", blob
        )
        venue_value = str((content.get("venue") or {}).get("value") or "")
        if "withdrawn" in venue_value.lower():
            continue
        if venue and not group and _short_venue(venue_value) not in {venue, _short_venue(venue)}:
            continue
        title = str((content.get("title") or {}).get("value") or "")
        arxiv_id = id_match.group(1) if id_match else ""
        if not arxiv_id and venue and title and lookups < limit:
            lookups += 1
            try:
                arxiv_id = _arxiv_id_for_title(title)
            except (urllib.error.URLError, TimeoutError):
                arxiv_id = ""
        year_match = re.search(r"(20\d{2})", venue_value)
        hit = _hit(
            arxiv_id,
            title,
            year_match.group(1) if year_match else "",
            str((content.get("abstract") or {}).get("value") or ""),
            venue_value,
        )
        if hit:
            results.append(hit)
        if len(results) >= limit:
            break
    return results


def _search_semantic_scholar(query: str, limit: int, venue: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, str] = {
        "query": query,
        "limit": str(min(limit, 100)),
        "fields": "title,year,venue,abstract,externalIds",
    }
    if venue and venue in S2_VENUE:
        params["venue"] = S2_VENUE[venue]
    payload = http_get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    )
    results = []
    for paper in payload.get("data") or []:
        external = paper.get("externalIds") or {}
        hit = _hit(
            str(external.get("ArXiv") or ""),
            paper.get("title") or "",
            str(paper.get("year") or ""),
            paper.get("abstract") or "",
            venue or str(paper.get("venue") or ""),
        )
        if hit:
            results.append(hit)
    return results


def _search_huggingface(query: str, limit: int) -> list[dict[str, Any]]:
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
        published = str(paper.get("publishedAt") or paper.get("published_at") or "")
        venue = ""
        for key in ("conference", "venue", "publishedIn"):
            if paper.get(key):
                venue = str(paper[key])
                break
        hit = _hit(
            str(paper.get("id") or paper.get("arxiv_id") or item.get("id") or ""),
            paper.get("title") or "",
            published,
            paper.get("summary") or paper.get("ai_summary") or "",
            venue,
        )
        if hit:
            results.append(hit)
    return results


def _merge_hits(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for hit in group:
            current = merged.get(hit["arxiv_id"])
            if current is None:
                merged[hit["arxiv_id"]] = dict(hit)
                continue
            for key in ("title", "year", "summary", "venue"):
                if not current.get(key) and hit.get(key):
                    current[key] = hit[key]
    return list(merged.values())


def search_papers(query: str, limit: int = 20, venue: str | None = None) -> list[dict[str, Any]]:
    """Search arXiv, OpenReview venues, Semantic Scholar, and Hugging Face."""
    hf_query = f"{venue} {query}".strip() if venue else query
    if venue:
        searches = (
            lambda: _search_openreview(query, limit, venue),
            lambda: _search_arxiv(query, limit, venue),
            lambda: _search_semantic_scholar(query, limit, venue),
            lambda: _search_huggingface(hf_query, limit),
        )
    else:
        searches = (
            lambda: _search_arxiv(query, limit),
            lambda: _search_openreview(query, limit),
            lambda: _search_semantic_scholar(query, limit),
            lambda: _search_huggingface(query, limit),
        )
    groups: list[list[dict[str, Any]]] = []
    for search in searches:
        try:
            groups.append(search())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LookupError, KeyError):
            continue
    return _merge_hits(groups)[:limit]


def cmd_search(args: argparse.Namespace) -> None:
    known = known_ids(args.log)
    query = args.query
    hits = search_papers(query, limit=args.limit, venue=args.venue)
    fresh = 0
    for hit in hits:
        seen = hit["arxiv_id"] in known
        marker = "OLD" if seen else "NEW"
        if not seen:
            fresh += 1
        print(f"[{marker}] {hit['arxiv_id']}\t{hit['year']}\t{hit.get('venue', '')}\t{hit['title']}")
        if args.log_new and not seen:
            append_row(
                args.log,
                {
                    "timestamp_utc": utc_now(),
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit["title"],
                    "venue": args.venue or hit.get("venue") or "",
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
            "url": latest.get("url", paper_url(arxiv_id)),
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
    hits = [
        hit
        for hit in search_papers(query, limit=args.limit, venue=args.venue)
        if hit["arxiv_id"] not in known
    ]
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

    search = sub.add_parser("search", help="search arXiv, conference venues, and HF papers")
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
