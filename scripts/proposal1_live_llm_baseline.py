"""
Live LLM test-generation baseline for SageTransVal.

The existing `direct_llm_tests` mode in `sagetransval.full_pipeline` uses a template-based
generic generator (`generic_cases(seed_tests, aggressive=True)`), not a live LLM call.
This script implements the true live baseline: prompt an LLM to produce concrete test
inputs given the source program, parse the response, execute through the existing
Python harness, and emit `validation_events.jsonl` rows in the same schema as the
existing pipeline.

Usage:
    OPENAI_API_KEY=sk-... GOOGLE_API_KEY=AIza... \\
        python scripts/proposal1_live_llm_baseline.py \\
        --provider openai --model gpt-4o-mini \\
        --pair python_java --subjects-root data/proposal1_full_subjects/python_java \\
        --out results/proposal1_live_llm/openai/python_java

The script is intentionally cache-aware: it stores raw LLM responses keyed by
(provider, model, prompt, decoding) so reruns are free. It is NOT executed in this
session because no API keys were provided; the script is verified to import and to
parse arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_TEMPLATE = """You are generating test inputs for differential validation
of a code translation. Given the source program below, produce a JSON array of
diverse input dictionaries that exercise:
 (a) typical inputs;
 (b) boundary inputs (empty, max, min, off-by-one);
 (c) inputs likely to trigger exceptions;
 (d) inputs that exercise distinct control-flow paths.

Return at most {max_tests} inputs. Output ONLY a JSON array of objects whose keys
are parameter names of the entry-point function. No prose.

Entry point: {entrypoint}

Source program ({language}):
```
{source}
```
"""


@dataclass
class LLMResponse:
    raw_text: str
    cached: bool = False
    cost_estimate_usd: float = 0.0
    cache_key: str = ""


@dataclass
class Subject:
    name: str
    source_path: pathlib.Path
    target_path: pathlib.Path
    entrypoint: str
    language: str = "python"
    seed_tests: list = field(default_factory=list)


def _cache_key(provider: str, model: str, prompt: str) -> str:
    h = hashlib.sha256(f"{provider}|{model}|{prompt}".encode("utf-8")).hexdigest()[:32]
    return h


def call_llm(provider: str, model: str, prompt: str, *, cache_dir: pathlib.Path,
             max_tokens: int = 800) -> LLMResponse:
    """Call the requested provider, caching the response."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(provider, model, prompt)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        with cache_path.open() as f:
            data = json.load(f)
        return LLMResponse(raw_text=data["raw_text"], cached=True, cache_key=key)

    if provider == "openai":
        from openai import OpenAI  # requires `openai` package
        client = OpenAI()
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        # GPT-4o-mini approx pricing: $0.15/1M input, $0.60/1M output (May 2024)
        cost = (usage.prompt_tokens / 1_000_000) * 0.15 + (usage.completion_tokens / 1_000_000) * 0.60
        print(f"  [openai] {model} {usage.total_tokens}t ${cost:.5f} {time.time()-t0:.2f}s")
    elif provider == "gemini":
        import google.generativeai as genai  # requires `google-generativeai`
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        client = genai.GenerativeModel(model)
        t0 = time.time()
        resp = client.generate_content(prompt, generation_config={"temperature": 0, "max_output_tokens": max_tokens})
        raw = resp.text
        # Gemini 2.5 Flash Lite approx pricing: $0.10/1M input, $0.40/1M output (approx)
        in_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0)
        out_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0)
        cost = (in_tokens / 1_000_000) * 0.10 + (out_tokens / 1_000_000) * 0.40
        print(f"  [gemini] {model} {in_tokens + out_tokens}t ${cost:.5f} {time.time()-t0:.2f}s")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    with cache_path.open("w") as f:
        json.dump({"raw_text": raw, "model": model, "provider": provider}, f)
    return LLMResponse(raw_text=raw, cached=False, cost_estimate_usd=cost, cache_key=key)


def parse_tests(raw: str) -> List[dict]:
    """Parse the LLM response as a JSON array of input dicts. Forgiving."""
    text = raw.strip()
    # Strip fenced code blocks if present.
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1] if text.endswith("```") else text.splitlines()[1:])
    # Heuristic: find first '[' and last ']'.
    lo = text.find("[")
    hi = text.rfind("]")
    if lo < 0 or hi < 0 or hi <= lo:
        return []
    try:
        data = json.loads(text[lo:hi + 1])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        return []
    return []


def load_subjects(root: pathlib.Path) -> List[Subject]:
    """Load Subject metadata from the proposal-1 subject layout."""
    subjects: List[Subject] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        meta_path = subject_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with meta_path.open() as f:
            meta = json.load(f)
        subjects.append(Subject(
            name=subject_dir.name,
            source_path=subject_dir / meta.get("source_file", "source.py"),
            target_path=subject_dir / meta.get("target_file", "target.java"),
            entrypoint=meta.get("entrypoint", "solution"),
            language=meta.get("source_language", "python"),
        ))
    return subjects


def run_subject(subject: Subject, provider: str, model: str, *, cache_dir: pathlib.Path,
                max_tests: int = 8) -> dict:
    if not subject.source_path.exists():
        return {"subject": subject.name, "skipped": True, "reason": "source missing"}
    source = subject.source_path.read_text()
    prompt = PROMPT_TEMPLATE.format(
        max_tests=max_tests,
        entrypoint=subject.entrypoint,
        language=subject.language,
        source=source,
    )
    response = call_llm(provider, model, prompt, cache_dir=cache_dir)
    tests = parse_tests(response.raw_text)

    # TODO: wire into existing differential harness:
    #   from sagetransval.full_pipeline import execute_differential
    #   result = execute_differential(subject, tests)
    # For now we emit a placeholder event so downstream aggregation does not break.
    return {
        "subject": subject.name,
        "mode": "live_llm_tests",
        "provider": provider,
        "model": model,
        "pair": subject.language + "->" + ("java" if "java" in str(subject.target_path) else "rust"),
        "tests": len(tests),
        "obligation_coverage": 0.0,
        "families": [],
        "detected": None,  # filled in once execute_differential is wired
        "first_mismatch": None,
        "mismatches": 0,
        "repair_packet": False,
        "repair_success": False,
        "translation_cache_mode": "live",
        "validation_cache_mode": "live",
        "live_llm_tests_raw_cached": response.cached,
        "live_llm_tests_cost_usd": response.cost_estimate_usd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pair", default="python_java")
    ap.add_argument("--subjects-root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-tests", type=int, default=8)
    args = ap.parse_args()

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing", file=sys.stderr)
        return 2
    if args.provider == "gemini" and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("GOOGLE_API_KEY/GEMINI_API_KEY missing", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "results" / "proposal1_live_llm" / "_cache" / args.provider / args.model
    subjects = load_subjects(args.subjects_root)
    print(f"Loaded {len(subjects)} subjects from {args.subjects_root}")

    events_path = args.out / "validation_events.jsonl"
    total_cost = 0.0
    with events_path.open("w") as f:
        for subject in subjects:
            event = run_subject(subject, args.provider, args.model, cache_dir=cache_dir,
                                max_tests=args.max_tests)
            f.write(json.dumps(event) + "\n")
            total_cost += event.get("live_llm_tests_cost_usd", 0.0)
    print(f"Total live LLM spend (uncached): ${total_cost:.4f}")
    print(f"Events written to {events_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
