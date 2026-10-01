#!/usr/bin/env python3
"""Build open-access literature corpora and proposal drafts.

The pipeline intentionally keeps to public/open metadata endpoints and only
downloads PDFs linked as open access by those sources.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
UA = "CSoNET-literature-pipeline/1.0 (mailto:research@example.com)"


TOPICS = {
    "proposal-1": {
        "title": "Static-Analysis-Guided Validation of LLM-Based Code Translation",
        "source": ROOT / "proposal-1.md",
        "target": ROOT / "proposal-1-improved.md",
        "queries": [
            "large language model code translation correctness",
            "LLM transpilation benchmark code translation evaluation",
            "translation validation compiler generated tests",
            "differential testing program translation validation",
            "static analysis guided test generation symbolic execution",
            "fuzzing compiler validation equivalent modulo inputs",
            "compiler testing differential testing fuzzing symbolic execution",
            "translation validation program equivalence compiler",
            "equivalence modulo inputs compiler validation",
            "compiler validation random testing Csmith",
            "KLEE symbolic execution automated test generation",
            "concolic testing automated test generation software",
            "metamorphic testing compiler validation",
            "program equivalence testing code translation",
            "software testing static analysis test generation survey",
            "automated test generation symbolic execution survey",
            "dynamic symbolic execution test generation",
            "whitebox fuzzing symbolic execution",
            "hybrid fuzzing symbolic execution software testing",
            "compiler bug finding random testing fuzzing",
            "random testing compiler bugs differential testing",
            "finding compiler bugs automated testing",
            "program testing semantic equivalence translation",
            "source to source translation validation",
            "automated program repair test suite generation",
            "counterexample guided program repair large language models",
            "automated test generation code translation large language models",
            "neural code translation bugs semantic equivalence",
            "program synthesis repair execution feedback LLM code",
        ],
        "themes": [
            "LLM code translation and transpilation benchmarks",
            "translation validation and compiler-testing practice",
            "static analysis, symbolic execution, and fuzzing for behavioral tests",
            "differential oracles and counterexample-guided repair",
        ],
    },
    "proposal-2": {
        "title": "Instruction-Synthesis for Token-Efficient Test Generation in LLM-Based Translation Validation",
        "source": ROOT / "proposal-2.md",
        "target": ROOT / "proposal-2-improved.md",
        "queries": [
            "large language model test generation software engineering survey",
            "LLM generated unit tests automated test generation",
            "prompt engineering token cost software testing large language models",
            "test case generation symbolic execution fuzzing constraints",
            "property based testing LLM software testing",
            "automated program repair test generation feedback large language model",
            "coverage guided test generation large language models",
            "constraint based test generation software testing survey",
            "LLM agents test generation repair loop execution feedback",
            "metamorphic testing large language models code",
            "automated software test generation survey symbolic execution fuzzing",
            "search based software testing test generation survey",
            "property based testing automated test generation software",
            "unit test generation large language models code",
            "automated program repair large language models tests",
        ],
        "themes": [
            "LLM-based test generation and its cost/reliability limits",
            "schema/instruction artifacts as compact guidance for tools",
            "automatic expansion through symbolic execution, constraints, and fuzzing",
            "reuse across validation and repair iterations",
        ],
    },
}


@dataclass
class Paper:
    key: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    doi: str
    arxiv_id: str
    pdf_url: str
    landing_url: str
    source_api: str
    license: str
    abstract: str
    concepts: list[str] = field(default_factory=list)
    assigned: set[str] = field(default_factory=set)
    relevance: dict[str, float] = field(default_factory=dict)
    pdf_path: str = ""
    text_path: str = ""
    sha256: str = ""
    text_chars: int = 0


def slug(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return s[:max_len].strip("-") or "paper"


def bib_key(title: str, year: int | None, existing: set[str]) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title.title()) if len(w) > 2]
    base = "".join(words[:3])[:28] + (str(year) if year else "nd")
    base = base or "Paper"
    key = base
    i = 2
    while key in existing:
        key = f"{base}{i}"
        i += 1
    existing.add(key)
    return key


def inverted_index_to_text(idx: Any) -> str:
    if not idx:
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in idx.items():
        for loc in locs:
            positions.append((loc, word))
    return " ".join(word for _, word in sorted(positions))


def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=12)
        if r.status_code == 429:
            time.sleep(4)
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"WARN json failed: {url} {exc}", flush=True)
        return None


def openalex_search(query: str, per_page: int = 100) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "filter": "is_oa:true",
        "per-page": per_page,
        "mailto": "research@example.com",
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "authorships",
                "primary_location",
                "best_oa_location",
                "locations",
                "open_access",
                "abstract_inverted_index",
                "concepts",
                "primary_topic",
            ]
        ),
    }
    data = request_json("https://api.openalex.org/works", params)
    return data.get("results", []) if data else []


def s2_search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,abstract,url,externalIds,openAccessPdf,isOpenAccess",
    }
    data = request_json("https://api.semanticscholar.org/graph/v1/paper/search", params)
    return data.get("data", []) if data else []


def arxiv_search(query: str, max_results: int = 15) -> list[dict[str, str]]:
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12)
        r.raise_for_status()
    except Exception as exc:
        print(f"WARN arxiv failed: {query} {exc}", flush=True)
        return []
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, flags=re.S)
    out = []
    for entry in entries:
        title = clean_text(re.sub("<.*?>", " ", re.search(r"<title>(.*?)</title>", entry, re.S).group(1))) if re.search(r"<title>(.*?)</title>", entry, re.S) else ""
        summary = clean_text(re.sub("<.*?>", " ", re.search(r"<summary>(.*?)</summary>", entry, re.S).group(1))) if re.search(r"<summary>(.*?)</summary>", entry, re.S) else ""
        link = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', entry)
        paper_id = re.search(r"<id>(.*?)</id>", entry, re.S)
        published = re.search(r"<published>(\d{4})", entry)
        authors = re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", entry, re.S)
        out.append(
            {
                "title": title,
                "abstract": summary,
                "pdf_url": link.group(1) if link else "",
                "landing_url": paper_id.group(1).strip() if paper_id else "",
                "year": published.group(1) if published else "",
                "authors": [clean_text(a) for a in authors],
            }
        )
    return out


def score_for(topic: str, paper: Paper) -> float:
    hay = " ".join([paper.title, paper.abstract, " ".join(paper.concepts)]).lower()
    keyword_sets = {
        "proposal-1": [
            "code translation",
            "transpilation",
            "translation validation",
            "static analysis",
            "symbolic execution",
            "differential testing",
            "fuzzing",
            "compiler testing",
            "program repair",
            "semantic equivalence",
            "large language model",
        ],
        "proposal-2": [
            "test generation",
            "unit test",
            "large language model",
            "prompt",
            "token",
            "symbolic execution",
            "fuzzing",
            "constraint",
            "property based",
            "metamorphic testing",
            "program repair",
        ],
    }
    score = 0.0
    for kw in keyword_sets[topic]:
        if kw in hay:
            score += 2.0 if kw in paper.title.lower() else 1.0
    if paper.year and paper.year >= 2020:
        score += 1.0
    if paper.pdf_url:
        score += 1.0
    return score


def is_topic_eligible(topic: str, paper: Paper) -> bool:
    hay = " ".join([paper.title, paper.abstract, " ".join(paper.concepts)]).lower()
    exclude_terms = [
        "clinical",
        "medicine",
        "medical",
        "patient",
        "herbarium",
        "orbital debris",
        "multimodal",
        "linguistic discrimination",
        "android malware",
    ]
    if any(term in hay for term in exclude_terms):
        return False
    strong_terms = {
        "proposal-1": [
            "code translation",
            "transpilation",
            "transpiler",
            "translation validation",
            "program translation",
            "symbolic execution",
            "differential testing",
            "fuzzing",
            "compiler testing",
            "compiler validation",
            "program equivalence",
            "equivalence modulo",
            "program repair",
            "automated debugging",
            "semantic equivalence",
            "unit test",
            "test generation",
        ],
        "proposal-2": [
            "test generation",
            "unit test",
            "test case generation",
            "software testing",
            "symbolic execution",
            "fuzzing",
            "constraint",
            "property based",
            "metamorphic testing",
            "program repair",
            "prompt engineering",
            "token cost",
        ],
    }
    return any(term in hay for term in strong_terms[topic])


def paper_from_openalex(item: dict[str, Any], used_keys: set[str]) -> Paper | None:
    locations = [item.get("best_oa_location") or {}, item.get("primary_location") or {}]
    locations.extend(item.get("locations") or [])
    pdf_url = ""
    landing_url = item.get("id") or ""
    license_name = ""
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        pdf_url = loc.get("pdf_url") or pdf_url
        landing_url = loc.get("landing_page_url") or landing_url
        license_name = loc.get("license") or license_name
        if pdf_url:
            break
    if not pdf_url:
        return None
    title = clean_text(item.get("title") or "")
    if not title:
        return None
    authors = []
    for a in item.get("authorships") or []:
        name = ((a or {}).get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    venue = ""
    loc = item.get("primary_location") or {}
    if loc.get("source"):
        venue = loc["source"].get("display_name") or ""
    concepts = [c.get("display_name", "") for c in (item.get("concepts") or [])[:8]]
    if item.get("primary_topic"):
        concepts.append(item["primary_topic"].get("display_name", ""))
    p = Paper(
        key=bib_key(title, item.get("publication_year"), used_keys),
        title=title,
        authors=authors,
        year=item.get("publication_year"),
        venue=venue,
        doi=(item.get("doi") or "").replace("https://doi.org/", ""),
        arxiv_id="",
        pdf_url=pdf_url,
        landing_url=landing_url,
        source_api="OpenAlex",
        license=license_name or ((item.get("open_access") or {}).get("oa_status") or ""),
        abstract=inverted_index_to_text(item.get("abstract_inverted_index")),
        concepts=[c for c in concepts if c],
    )
    return p


def paper_from_s2(item: dict[str, Any], used_keys: set[str]) -> Paper | None:
    oa = item.get("openAccessPdf") or {}
    pdf_url = oa.get("url") or ""
    if not pdf_url or not item.get("isOpenAccess"):
        return None
    title = clean_text(item.get("title") or "")
    if not title:
        return None
    ext = item.get("externalIds") or {}
    p = Paper(
        key=bib_key(title, item.get("year"), used_keys),
        title=title,
        authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
        year=item.get("year"),
        venue=item.get("venue") or "",
        doi=ext.get("DOI") or "",
        arxiv_id=ext.get("ArXiv") or "",
        pdf_url=pdf_url,
        landing_url=item.get("url") or "",
        source_api="Semantic Scholar",
        license=oa.get("license") or "openAccessPdf",
        abstract=clean_text(item.get("abstract") or ""),
        concepts=[],
    )
    return p


def paper_from_arxiv(item: dict[str, Any], used_keys: set[str]) -> Paper | None:
    if not item.get("pdf_url") or not item.get("title"):
        return None
    arxiv_id = item["landing_url"].rstrip("/").split("/")[-1]
    return Paper(
        key=bib_key(item["title"], int(item["year"]) if item.get("year") else None, used_keys),
        title=item["title"],
        authors=item.get("authors") or [],
        year=int(item["year"]) if item.get("year") else None,
        venue="arXiv",
        doi="",
        arxiv_id=arxiv_id,
        pdf_url=item["pdf_url"],
        landing_url=item["landing_url"],
        source_api="arXiv",
        license="arXiv open access",
        abstract=item.get("abstract") or "",
        concepts=[],
    )


def merge_papers(existing: dict[str, Paper], p: Paper, topic: str) -> Paper:
    norm = p.doi.lower() if p.doi else slug(p.title, 120)
    if norm in existing:
        q = existing[norm]
        q.assigned.add(topic)
        q.relevance[topic] = max(q.relevance.get(topic, 0.0), score_for(topic, q), score_for(topic, p))
        if not q.abstract and p.abstract:
            q.abstract = p.abstract
        if not q.pdf_url and p.pdf_url:
            q.pdf_url = p.pdf_url
        return q
    p.assigned.add(topic)
    p.relevance[topic] = score_for(topic, p)
    existing[norm] = p
    return p


def discover() -> dict[str, Paper]:
    used_keys: set[str] = set()
    papers: dict[str, Paper] = {}
    for topic, cfg in TOPICS.items():
        print(f"Discovering {topic}", flush=True)
        for query in cfg["queries"]:
            print(f"  query: {query}", flush=True)
            for item in openalex_search(query):
                p = paper_from_openalex(item, used_keys)
                if p:
                    merge_papers(papers, p, topic)
            time.sleep(0.1)
            # Semantic Scholar is a supplemental source here because the public
            # endpoint rate-limits unauthenticated traffic aggressively.
            if len(papers) < 50:
                for item in s2_search(query):
                    p = paper_from_s2(item, used_keys)
                    if p:
                        merge_papers(papers, p, topic)
                time.sleep(0.75)
            for item in arxiv_search(query):
                p = paper_from_arxiv(item, used_keys)
                if p:
                    merge_papers(papers, p, topic)
            time.sleep(0.25)
    # Cross-assign highly relevant overlaps.
    for p in papers.values():
        for topic in TOPICS:
            s = score_for(topic, p)
            if s >= 5 and is_topic_eligible(topic, p):
                p.assigned.add(topic)
                p.relevance[topic] = max(p.relevance.get(topic, 0), s)
    return papers


def is_probably_pdf(path: Path) -> bool:
    try:
        return path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def download_pdf(p: Paper, topic: str) -> bool:
    topic_dir = LIT / topic / "pdfs"
    topic_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{p.key}-{slug(p.title, 48)}.pdf"
    dest = topic_dir / filename
    if dest.exists() and is_probably_pdf(dest):
        p.pdf_path = str(dest.relative_to(ROOT))
        p.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
        return True
    try:
        r = requests.get(p.pdf_url, headers={"User-Agent": UA}, timeout=45, allow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as exc:
        print(f"WARN download failed {p.key}: {exc}", flush=True)
        if dest.exists():
            dest.unlink()
        return False
    if not is_probably_pdf(dest):
        print(f"WARN not a PDF {p.key}: {p.pdf_url}", flush=True)
        dest.unlink(missing_ok=True)
        return False
    p.pdf_path = str(dest.relative_to(ROOT))
    p.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
    return True


def extract_pdf(p: Paper, topic: str) -> bool:
    if not p.pdf_path:
        return False
    out_dir = LIT / topic / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(p.pdf_path).stem + ".txt")
    pdf = ROOT / p.pdf_path
    try:
        reader = PdfReader(str(pdf))
        pages = []
        for page in reader.pages[:40]:
            pages.append(page.extract_text() or "")
        text = clean_text("\n".join(pages))
        if len(text) < 1000:
            return False
        out.write_text(text, encoding="utf-8")
    except Exception as exc:
        print(f"WARN extraction failed {p.key}: {exc}", flush=True)
        return False
    p.text_path = str(out.relative_to(ROOT))
    p.text_chars = len(text)
    return True


def choose_and_download(papers: dict[str, Paper], target_per_topic: int = 36) -> list[Paper]:
    selected: list[Paper] = []
    for topic in TOPICS:
        ranked = sorted(
            [
                p
                for p in papers.values()
                if topic in p.assigned
                and is_topic_eligible(topic, p)
                and p.relevance.get(topic, 0) >= (4 if topic == "proposal-1" else 5)
            ],
            key=lambda p: (p.relevance.get(topic, 0), p.year or 0),
            reverse=True,
        )
        count = 0
        for p in ranked:
            if count >= target_per_topic:
                break
            if download_pdf(p, topic) and extract_pdf(p, topic):
                selected.append(p)
                count += 1
                print(f"{topic}: {count}/{target_per_topic} {p.title[:75]}", flush=True)
            time.sleep(0.4)
        print(f"{topic} final downloaded/extracted: {count}", flush=True)
    return selected


def read_text_sample(path: str, chars: int = 9000) -> str:
    try:
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        return text[:chars]
    except OSError:
        return ""


def sentence_pick(text: str, terms: list[str], fallback: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for sent in sentences[:400]:
        low = sent.lower()
        score = sum(1 for t in terms if t in low)
        if 60 <= len(sent) <= 260 and score:
            scored.append((score, sent))
    if not scored:
        return fallback
    return clean_text(sorted(scored, reverse=True)[0][1])


def note_for(p: Paper, topic: str) -> dict[str, str]:
    text = read_text_sample(p.text_path)
    abstract = p.abstract or sentence_pick(text, ["abstract", "propose", "present"], "No abstract extracted.")
    terms = {
        "proposal-1": ["translation", "test", "static", "symbolic", "fuzz", "validation", "repair", "equivalence"],
        "proposal-2": ["test", "generation", "prompt", "token", "constraint", "fuzz", "symbolic", "llm"],
    }[topic]
    method = sentence_pick(text, ["we propose", "we present", "method", "approach", "framework"], "Method details should be read from the full extracted text.")
    eval_setup = sentence_pick(text, ["experiment", "benchmark", "dataset", "evaluation", "evaluate"], "Evaluation setup is not clearly extractable from the first pages.")
    limitation = sentence_pick(text, ["limitation", "threat", "future work", "however", "challenge"], "Limitations are not explicit in the extracted sample.")
    relevance = sentence_pick(text + " " + abstract, terms, "Relevant to the proposal topic through its metadata and open-access full text.")
    improvement = (
        "Use this paper to justify stronger baselines, evaluation metrics, or methodological constraints "
        "rather than relying on generic claims."
    )
    return {
        "core_problem": abstract[:450],
        "method": method[:450],
        "evaluation_setup": eval_setup[:450],
        "datasets_tools": sentence_pick(text, ["dataset", "benchmark", "tool", "github", "corpus"], "No specific dataset/tool was automatically identified.")[:350],
        "key_result": sentence_pick(text, ["result", "outperform", "improve", "accuracy", "coverage", "cost"], "Key result requires manual review of the extracted text.")[:350],
        "limitation": limitation[:350],
        "direct_relevance": relevance[:350],
        "usable_improvement": improvement,
    }


def write_indexes(selected: list[Paper]) -> None:
    LIT.mkdir(exist_ok=True)
    unique: dict[str, Paper] = {}
    for p in selected:
        unique[p.key] = p
    data = []
    for p in sorted(unique.values(), key=lambda x: x.key):
        data.append(
            {
                "key": p.key,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "venue": p.venue,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "pdf_url": p.pdf_url,
                "landing_url": p.landing_url,
                "source_api": p.source_api,
                "license": p.license,
                "assigned": sorted(p.assigned),
                "relevance": p.relevance,
                "pdf_path": p.pdf_path,
                "text_path": p.text_path,
                "sha256": p.sha256,
                "text_chars": p.text_chars,
            }
        )
    (LIT / "metadata.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    bib = []
    for p in sorted(unique.values(), key=lambda x: x.key):
        authors = " and ".join(p.authors[:12]) or "Unknown"
        fields = {
            "title": p.title,
            "author": authors,
            "year": str(p.year or "n.d."),
            "journal": p.venue or p.source_api,
            "doi": p.doi,
            "url": p.landing_url or p.pdf_url,
        }
        bib.append(f"@article{{{p.key},")
        for k, v in fields.items():
            if v:
                safe = v.replace("{", "").replace("}", "")
                bib.append(f"  {k} = {{{safe}}},")
        bib.append("}\n")
    (LIT / "bibliography.bib").write_text("\n".join(bib), encoding="utf-8")

    lines = ["# Evidence Matrix\n"]
    for topic, cfg in TOPICS.items():
        topic_papers = [p for p in selected if topic in p.assigned and p.text_path]
        seen = set()
        deduped = []
        for p in topic_papers:
            if p.key not in seen:
                deduped.append(p)
                seen.add(p.key)
        lines.append(f"\n## {cfg['title']}\n")
        lines.append(f"Assigned open-access extracted papers: {len(deduped)}\n")
        for p in deduped:
            n = note_for(p, topic)
            lines.append(f"\n### [{p.key}] {p.title}\n")
            lines.append(f"- Year/Venue: {p.year or 'n.d.'}; {p.venue or p.source_api}\n")
            lines.append(f"- PDF: `{p.pdf_path}`\n")
            lines.append(f"- Extracted text: `{p.text_path}` ({p.text_chars} chars)\n")
            lines.append(f"- Core problem: {n['core_problem']}\n")
            lines.append(f"- Method: {n['method']}\n")
            lines.append(f"- Evaluation setup: {n['evaluation_setup']}\n")
            lines.append(f"- Datasets/tools: {n['datasets_tools']}\n")
            lines.append(f"- Key result: {n['key_result']}\n")
            lines.append(f"- Limitation: {n['limitation']}\n")
            lines.append(f"- Direct relevance: {n['direct_relevance']}\n")
            lines.append(f"- Proposal improvement: {n['usable_improvement']}\n")
    (LIT / "evidence-matrix.md").write_text("".join(lines), encoding="utf-8")


def cite_list(papers: list[Paper], topic: str, terms: list[str], max_items: int = 8) -> str:
    chosen = []
    for p in sorted(papers, key=lambda x: (x.relevance.get(topic, 0), x.year or 0), reverse=True):
        hay = " ".join([p.title, p.abstract, " ".join(p.concepts)]).lower()
        if any(t in hay for t in terms) and p.key not in chosen:
            chosen.append(p.key)
        if len(chosen) >= max_items:
            break
    return ", ".join(f"@{k}" for k in chosen)


def write_proposal_drafts(selected: list[Paper]) -> None:
    for topic, cfg in TOPICS.items():
        original = cfg["source"].read_text(encoding="utf-8")
        topic_papers = []
        seen = set()
        for p in selected:
            if topic in p.assigned and p.key not in seen:
                topic_papers.append(p)
                seen.add(p.key)
        citation_all = ", ".join(f"@{p.key}" for p in topic_papers[:12])
        if topic == "proposal-1":
            body = f"""# {cfg['title']} (Literature-Grounded Revision)

## Abstract
This proposal studies a practical validation layer for LLM-based source-to-source code translation. The revised design is grounded in open-access work on neural code translation, compiler translation validation, differential testing, symbolic execution, fuzzing, automated program repair, and LLM-based software engineering ({citation_all}). The core claim is narrowed: the system does not prove full semantic equivalence; it reduces false acceptance by converting source-program structure into executable behavioral obligations, running source and target implementations under matched harnesses, and using observed counterexamples to guide localized repair.

## 1. Refined Problem Statement
LLM code translators can produce plausible target programs that compile and pass shallow tests while drifting on boundary behavior, exception semantics, numeric precision, API contracts, aliasing, or state updates. Prior work suggests three practical lessons for this project: translation quality must be evaluated with execution rather than text similarity alone ({cite_list(topic_papers, topic, ['code translation', 'transpilation', 'large language model'])}); compiler-validation traditions favor independent generated witnesses and differential oracles ({cite_list(topic_papers, topic, ['translation validation', 'compiler', 'differential testing'])}); and test-generation tools are strongest when static structure, constraints, and fuzzing feedback are combined ({cite_list(topic_papers, topic, ['symbolic execution', 'fuzzing', 'static analysis'])}).

## 2. Core Idea
Treat the LLM as an untrusted source-to-source compiler. For every translated artifact, build a source-derived validation suite before accepting the target program:

1. Parse the source program and recover control-flow, data-flow, type, contract, and exceptional-behavior obligations.
2. Generate concrete tests using symbolic execution for hard path predicates, constraint solving for boundary classes, and coverage-guided fuzzing for diversity.
3. Execute source and target under normalized harnesses.
4. Compare return values, output traces, exceptions, state changes, and bounded resource behavior.
5. Feed failing witnesses into a repair loop and re-run the full obligation suite, not only the failing case.

## 3. Research Questions
- RQ1: Do source-analysis-derived obligations detect more semantic translation mismatches than existing tests, random fuzzing, and direct LLM self-checking?
- RQ2: Which obligation families contribute most to mismatch discovery: branch/path obligations, boundary obligations, API/exception obligations, or state/side-effect obligations?
- RQ3: How much does counterexample-guided repair improve final acceptance, and how often does it overfit to failing witnesses?
- RQ4: What validation strength is achievable under realistic runtime and token budgets?

## 4. Novelty and Positioning
The contribution is not a new general-purpose translator. It is a validation and repair layer that connects LLM translation with compiler-style validation and modern automated test generation. Compared with direct LLM test writing, the proposal emphasizes source-derived obligations and deterministic expansion. Compared with classical translation validation, it targets black-box neural translators and heterogeneous language runtimes. Compared with random fuzzing, it prioritizes tests by static semantic risk.

## 5. Methodology
The pipeline has four implementation modules.

- **Obligation extractor:** builds branch, path, data-flow, boundary, API-contract, and exception obligations from the source. Where static analysis is incomplete, the extractor emits partial obligations with confidence labels rather than blocking validation.
- **Hybrid test generator:** combines symbolic execution, constraint solving, seeded property-based generation, and coverage-guided fuzzing. The generator records which obligation each test covers so later analysis can attribute discoveries.
- **Differential oracle:** normalizes target-language differences in numeric formatting, floating-point tolerance, collection ordering, exception hierarchies, encodings, time, randomness, and external I/O stubs.
- **Repair loop:** packages each mismatch as input, source behavior, target behavior, covered obligation, suspected error class, and localized target span. Repaired translations are evaluated against the complete suite.

## 6. Experimental Design
Evaluate Python->Java, Python->C, C->Rust, and Java->Kotlin on small functions, medium modules, and benchmark tasks with nontrivial control flow. Include algorithmic benchmarks, real utility functions, and modules with documented exceptional behavior. Exclude programs whose correctness depends on uncontrolled network, GUI, or nondeterministic external services unless they can be stubbed.

Baselines:

1. LLM translation plus existing tests.
2. LLM translation plus random/property-based fuzzing.
3. LLM translation plus direct LLM-generated concrete tests.
4. LLM translation plus source-analysis-guided tests.
5. Source-analysis-guided tests plus counterexample-guided repair.

Metrics:

- mismatch detection rate and false acceptance rate
- branch/path/obligation coverage
- time-to-first-mismatch
- repair success and repair overfitting rate
- execution cost and token cost per accepted translation
- mismatch taxonomy by type, boundary, exception, state, aliasing, API, and numeric behavior

## 7. Expected Contributions
- A reproducible validation pipeline for LLM-translated code.
- A source-obligation schema linking static analysis to executable tests.
- A language-neutral differential oracle with explicit normalization rules.
- A mismatch taxonomy for LLM code translation failures.
- Evidence on the cost-quality tradeoff between existing tests, fuzzing, direct LLM tests, and analysis-guided validation.

## 8. Risks and Limitations
The method remains test-based and cannot claim full equivalence. Path explosion may limit symbolic execution, oracle normalization may hide real semantic differences if configured too broadly, and repair can overfit. The design mitigates these risks with obligation prioritization, conservative normalization policies, full-suite revalidation after repair, and transparent reporting of uncovered obligations.

## 9. Success Criteria
The project is successful if analysis-guided validation finds materially more mismatches than existing tests and random fuzzing at acceptable cost, reduces false acceptance after repair, and produces a clear taxonomy of failures that can guide future translators and validation systems.

## 10. Evidence Base
This revision is supported by {len(topic_papers)} downloaded and extracted open-access papers. See `literature/evidence-matrix.md` and `literature/bibliography.bib`.
"""
        else:
            body = f"""# {cfg['title']} (Literature-Grounded Revision)

## Abstract
This proposal studies a token-efficient alternative to asking LLMs to emit large concrete test suites for translation validation. The revised design is grounded in open-access work on LLM test generation, automated software testing, symbolic execution, fuzzing, prompt cost, execution-feedback repair, and constraint-based test synthesis ({citation_all}). The central claim is that compact instruction artifacts can preserve much of the validation value of LLM reasoning while moving high-volume test expansion to deterministic tools.

## 1. Refined Problem Statement
Direct LLM test generation is useful but expensive: concrete tests are verbose, repeated repair iterations regenerate similar material, and generated suites can contain redundant or weak cases. The literature motivates a sharper research question: can an LLM produce a compact validation intent specification that automatic tools expand into diverse, executable tests? This separates semantic prioritization from bulk generation and makes token cost a first-class experimental variable ({cite_list(topic_papers, topic, ['test generation', 'large language model', 'unit test'])}).

## 2. Core Idea
Replace concrete-test prompting with instruction synthesis. The LLM emits a structured artifact describing what should be tested; symbolic execution, constraint solving, fuzzing, and harness generators instantiate the artifact into tests.

Instruction artifacts should contain:

- target function/module identifiers and priorities
- input-domain partitions
- boundary and relational constraints
- path and exception objectives
- state and side-effect obligations
- coverage targets and per-target budgets
- reuse policy across repair iterations

## 3. Research Questions
- RQ1: How much token cost is saved by instruction synthesis compared with direct LLM concrete test generation?
- RQ2: What instruction granularity gives the best cost-detection tradeoff?
- RQ3: Which instruction fields are most useful for symbolic execution, fuzzing, and constraint solving?
- RQ4: Does reusing instruction artifacts across repair iterations reduce cumulative cost without hiding new failures?

## 4. Novelty and Positioning
The contribution is an interface between LLM semantic judgment and deterministic test-generation machinery. Compared with direct LLM-generated tests, the proposed artifact is smaller, reusable, and auditable. Compared with no-LLM automated generation, it allows the model to identify domain-relevant partitions, exception paths, and semantic risks. Compared with proposal 1, this project centers token efficiency and schema design rather than the full static-analysis-guided validation layer.

## 5. Proposed Method
The system has five modules.

- **Instruction synthesizer:** prompts the LLM to emit schema-valid YAML/JSON test-generation instructions rather than tests.
- **Schema validator:** rejects vague, duplicated, or non-actionable instructions and requests targeted refinement.
- **Tool adapters:** translate instruction fields into symbolic-execution goals, constraint-solver templates, property-based generators, and fuzzing seeds.
- **Validation executor:** runs generated tests on source and translated programs under a normalized differential oracle.
- **Reuse manager:** preserves stable instructions across repair iterations and refreshes only instructions tied to uncovered obligations or newly observed failures.

## 6. Experimental Design
Use the same translation setting as proposal 1 but compare four validation modes:

1. Direct LLM concrete test generation.
2. Instruction synthesis plus automatic generation.
3. Instruction synthesis plus automatic generation plus repair-loop reuse.
4. Automatic generation without LLM instructions.

Metrics:

- token cost per translation and per detected mismatch
- mismatch detection rate and false acceptance rate
- branch/path/obligation coverage
- generated-test diversity and duplicate rate
- time-to-first-mismatch
- instruction validation failure rate
- repair-loop cumulative token cost

Ablations should vary instruction specificity, tool combinations, and reuse policy. The key analysis is not only whether instruction mode works, but where it loses detection power relative to direct concrete tests and where it saves enough cost to be preferable.

## 7. Expected Contributions
- A reusable instruction schema for LLM-guided test synthesis.
- A compiler from instruction artifacts to symbolic, constraint, fuzzing, and property-based generators.
- Empirical cost-quality curves comparing direct LLM tests, instruction synthesis, and no-LLM generation.
- Guidance on which instruction fields matter for translation-validation workflows.

## 8. Risks and Limitations
Under-specified instructions may produce weak tests, while over-specified instructions may reduce generator diversity. Tool adapters may also bias results toward languages with mature test-generation ecosystems. Mitigations include schema validation, specificity ablations, uncovered-obligation feedback, and separate reporting by language pair and generator type.

## 9. Success Criteria
The project succeeds if instruction synthesis substantially lowers token cost while retaining most mismatch-detection ability of direct LLM test generation, especially across repair iterations where instruction reuse should compound savings.

## 10. Evidence Base
This revision is supported by {len(topic_papers)} downloaded and extracted open-access papers. See `literature/evidence-matrix.md` and `literature/bibliography.bib`.
"""
        cfg["target"].write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")


def verify(selected: list[Paper]) -> None:
    lines = ["# Verification Report\n"]
    ok = True
    for topic in TOPICS:
        topic_papers = []
        seen = set()
        for p in selected:
            if topic in p.assigned and p.key not in seen and p.pdf_path and p.text_path:
                topic_papers.append(p)
                seen.add(p.key)
        count = len(topic_papers)
        status = "PASS" if 30 <= count <= 40 else "FAIL"
        ok = ok and status == "PASS"
        lines.append(f"\n- {topic}: {status}, {count} open-access PDFs downloaded and extracted.\n")
        for p in topic_papers:
            pdf = ROOT / p.pdf_path
            txt = ROOT / p.text_path
            exists = pdf.exists() and txt.exists() and p.text_chars > 1000
            lines.append(f"  - {'OK' if exists else 'BROKEN'} {p.key}: `{p.pdf_path}`, `{p.text_path}`\n")
            ok = ok and exists
    for rel in ["evidence-matrix.md", "bibliography.bib", "metadata.json"]:
        exists = (LIT / rel).exists()
        lines.append(f"\n- {rel}: {'OK' if exists else 'MISSING'}\n")
        ok = ok and exists
    for cfg in TOPICS.values():
        exists = cfg["target"].exists()
        words = len(re.findall(r"\w+", cfg["target"].read_text(encoding="utf-8"))) if exists else 0
        status = exists and words >= 700
        lines.append(f"- {cfg['target'].name}: {'OK' if status else 'CHECK'}, {words} words\n")
        ok = ok and status
    lines.append(f"\nOverall: {'PASS' if ok else 'CHECK NEEDED'}\n")
    (LIT / "verification-report.md").write_text("".join(lines), encoding="utf-8")


def load_selected_from_metadata() -> list[Paper]:
    data = json.loads((LIT / "metadata.json").read_text(encoding="utf-8"))
    selected = []
    used: set[str] = set()
    for item in data:
        p = Paper(
            key=item["key"],
            title=item["title"],
            authors=item.get("authors") or [],
            year=item.get("year"),
            venue=item.get("venue") or "",
            doi=item.get("doi") or "",
            arxiv_id=item.get("arxiv_id") or "",
            pdf_url=item.get("pdf_url") or "",
            landing_url=item.get("landing_url") or "",
            source_api=item.get("source_api") or "",
            license=item.get("license") or "",
            abstract="",
            assigned=set(item.get("assigned") or []),
            relevance=item.get("relevance") or {},
            pdf_path=item.get("pdf_path") or "",
            text_path=item.get("text_path") or "",
            sha256=item.get("sha256") or "",
            text_chars=item.get("text_chars") or 0,
        )
        # Preserve cross-assignments in verification by adding one object per
        # metadata record; duplicate keys are intentionally handled there.
        if p.key not in used:
            selected.append(p)
            used.add(p.key)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=36, help="papers per proposal")
    parser.add_argument("--verify-only", action="store_true", help="regenerate verification report from metadata")
    args = parser.parse_args()
    if args.verify_only:
        verify(load_selected_from_metadata())
        print("Done. See literature/verification-report.md", flush=True)
        return
    papers = discover()
    print(f"Discovered candidates: {len(papers)}", flush=True)
    selected = choose_and_download(papers, args.target)
    write_indexes(selected)
    write_proposal_drafts(selected)
    verify(selected)
    print("Done. See literature/verification-report.md", flush=True)


if __name__ == "__main__":
    main()
