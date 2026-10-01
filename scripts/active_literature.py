#!/usr/bin/env python3
"""Collect active/adaptive validation literature for proposal 2."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature" / "proposal-2-active"
UA = "CSoNET-active-validation-literature/1.0"


SEEDS = [
    {
        "key": "EcofuzzAdaptiveEnergy2020",
        "title": "EcoFuzz: Adaptive Energy-Saving Greybox Fuzzing as a Variant of the Adversarial Multi-Armed Bandit",
        "url": "https://www.usenix.org/system/files/sec20-yue.pdf",
        "year": 2020,
        "venue": "USENIX Security",
        "relevance": "Frames fuzzing resource allocation as an adversarial multi-armed bandit, directly supporting budgeted validation planning.",
    },
    {
        "key": "TSchedulerBandit2023",
        "title": "T-Scheduler: A Time-Aware Seed Scheduler Based on Multi-Armed Bandit for Greybox Fuzzing",
        "url": "https://arxiv.org/pdf/2312.04749",
        "year": 2023,
        "venue": "arXiv",
        "relevance": "Uses bandit seed scheduling for fuzzing, a close analogue for choosing validation actions under budget.",
    },
    {
        "key": "MabfuzzBandit2023",
        "title": "MABFuzz: Multi-Armed Bandit Algorithms for Processor Fuzzing",
        "url": "https://arxiv.org/pdf/2311.14594",
        "year": 2023,
        "venue": "arXiv",
        "relevance": "Shows how bandit policies can allocate fuzzing effort across action choices.",
    },
    {
        "key": "MobfuzzAdaptive2024",
        "title": "MobFuzz: Adaptive Multi-objective Optimization in Gray-box Fuzzing",
        "url": "https://arxiv.org/pdf/2401.15956",
        "year": 2024,
        "venue": "arXiv",
        "relevance": "Motivates multi-objective planning over coverage, failure discovery, and cost.",
    },
    {
        "key": "ReinforcementFeedbackUnit2024",
        "title": "Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation",
        "url": "https://arxiv.org/pdf/2412.14308",
        "year": 2024,
        "venue": "arXiv",
        "relevance": "Connects test generation with reward design and automatic feedback.",
    },
]

QUERIES = [
    "adaptive random testing software testing survey",
    "active learning software test generation",
    "adaptive test generation software testing",
    "multi armed bandit fuzzing seed scheduling",
    "reinforcement learning test generation software",
    "budgeted test generation software engineering",
    "counterexample guided test generation software",
    "active learning fuzzing software testing",
]


@dataclass
class Paper:
    key: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    url: str
    pdf_url: str
    source: str
    relevance: str
    pdf_path: str = ""
    text_path: str = ""
    sha256: str = ""
    text_chars: int = 0


def slug(s: str, max_len: int = 70) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")[:max_len].strip("-") or "paper"


def key_for(title: str, year: int | None, used: set[str]) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title.title()) if len(w) > 2]
    base = "".join(words[:3])[:30] + (str(year) if year else "nd")
    key = base or "Paper"
    i = 2
    while key in used:
        key = f"{base}{i}"
        i += 1
    used.add(key)
    return key


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()


def arxiv_search(query: str, max_results: int = 12) -> list[Paper]:
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
    except Exception:
        return []
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, flags=re.S)
    papers = []
    used: set[str] = set()
    for e in entries:
        title_m = re.search(r"<title>(.*?)</title>", e, re.S)
        link_m = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', e)
        id_m = re.search(r"<id>(.*?)</id>", e, re.S)
        year_m = re.search(r"<published>(\d{4})", e)
        authors = [clean(a) for a in re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", e, re.S)]
        if not title_m or not link_m:
            continue
        title = clean(re.sub("<.*?>", " ", title_m.group(1)))
        hay = title.lower()
        if not any(t in hay for t in ["fuzz", "test", "testing", "bandit", "reinforcement", "adaptive", "active"]):
            continue
        year = int(year_m.group(1)) if year_m else None
        papers.append(Paper(key_for(title, year, used), title, authors, year, "arXiv", id_m.group(1).strip() if id_m else link_m.group(1), link_m.group(1), "arXiv", "Relevant to active/adaptive test generation or budgeted validation."))
    return papers


def openalex_search(query: str, per_page: int = 25) -> list[Paper]:
    params = {
        "search": query,
        "filter": "is_oa:true",
        "per-page": per_page,
        "mailto": "research@example.com",
        "select": "id,doi,title,publication_year,authorships,primary_location,best_oa_location,locations,open_access,abstract_inverted_index",
    }
    try:
        r = requests.get("https://api.openalex.org/works", params=params, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    out = []
    used: set[str] = set()
    for item in data.get("results", []):
        title = clean(item.get("title") or "")
        if not title:
            continue
        hay = title.lower()
        if not any(t in hay for t in ["fuzz", "test", "testing", "bandit", "reinforcement", "adaptive", "active"]):
            continue
        pdf = ""
        landing = item.get("id") or ""
        for loc in [item.get("best_oa_location") or {}, item.get("primary_location") or {}, *(item.get("locations") or [])]:
            if isinstance(loc, dict):
                pdf = loc.get("pdf_url") or pdf
                landing = loc.get("landing_page_url") or landing
            if pdf:
                break
        if not pdf:
            continue
        authors = [((a or {}).get("author") or {}).get("display_name") for a in item.get("authorships", [])]
        authors = [a for a in authors if a]
        venue = ""
        primary = item.get("primary_location") or {}
        if primary.get("source"):
            venue = primary["source"].get("display_name") or ""
        out.append(Paper(key_for(title, item.get("publication_year"), used), title, authors, item.get("publication_year"), venue or "OpenAlex", landing, pdf, "OpenAlex", "Relevant to active/adaptive test generation, fuzzing, or budget-aware validation."))
    return out


def is_pdf(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:5] == b"%PDF-"


def download(p: Paper) -> bool:
    dest = OUT / "pdfs" / f"{p.key}-{slug(p.title)}.pdf"
    if not dest.exists():
        try:
            r = requests.get(p.pdf_url, headers={"User-Agent": UA}, timeout=45, allow_redirects=True)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as exc:
            print(f"WARN download {p.key}: {exc}")
            dest.unlink(missing_ok=True)
            return False
    if not is_pdf(dest):
        dest.unlink(missing_ok=True)
        return False
    p.pdf_path = str(dest.relative_to(ROOT))
    p.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
    return True


def extract(p: Paper) -> bool:
    pdf = ROOT / p.pdf_path
    out = OUT / "extracted" / (pdf.stem + ".txt")
    try:
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])
        text = clean(text)
    except Exception as exc:
        print(f"WARN extract {p.key}: {exc}")
        return False
    if len(text) < 1000:
        return False
    out.write_text(text, encoding="utf-8")
    p.text_path = str(out.relative_to(ROOT))
    p.text_chars = len(text)
    return True


def main() -> None:
    (OUT / "pdfs").mkdir(parents=True, exist_ok=True)
    (OUT / "extracted").mkdir(parents=True, exist_ok=True)
    candidates: list[Paper] = []
    for s in SEEDS:
        candidates.append(Paper(s["key"], s["title"], [], s["year"], s["venue"], s["url"], s["url"], "seed", s["relevance"]))
    for q in QUERIES:
        print(f"query {q}", flush=True)
        candidates.extend(openalex_search(q))
        time.sleep(0.2)
        candidates.extend(arxiv_search(q))
        time.sleep(0.5)
    seen: set[str] = set()
    selected: list[Paper] = []
    for p in candidates:
        norm = re.sub(r"\W+", "", p.title.lower())
        if norm in seen:
            continue
        seen.add(norm)
        if download(p) and extract(p):
            selected.append(p)
            print(f"{len(selected)} {p.title[:80]}", flush=True)
        if len(selected) >= 16:
            break
    data = [p.__dict__ for p in selected]
    (OUT / "metadata.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Active Validation Related Work\n\n"]
    for p in selected:
        lines.append(f"## [{p.key}] {p.title}\n")
        lines.append(f"- Year/Venue: {p.year or 'n.d.'}; {p.venue}\n")
        lines.append(f"- Source: {p.source}; {p.url}\n")
        lines.append(f"- PDF: `{p.pdf_path}`\n")
        lines.append(f"- Extracted text: `{p.text_path}` ({p.text_chars} chars)\n")
        lines.append(f"- Relevance: {p.relevance}\n\n")
    (OUT / "evidence.md").write_text("".join(lines), encoding="utf-8")
    bib = []
    for p in selected:
        authors = " and ".join(p.authors[:12]) if p.authors else "Unknown"
        title = p.title.replace("&", "and").replace("{", "").replace("}", "")
        bib.append(f"@article{{{p.key},")
        bib.append(f"  title = {{{title}}},")
        bib.append(f"  author = {{{authors}}},")
        if p.year:
            bib.append(f"  year = {{{p.year}}},")
        bib.append(f"  journal = {{{p.venue or p.source}}},")
        bib.append(f"  url = {{{p.url}}},")
        bib.append("}\n")
    (OUT / "active-references.bib").write_text("\n".join(bib), encoding="utf-8")
    print(f"selected {len(selected)}")


if __name__ == "__main__":
    main()
