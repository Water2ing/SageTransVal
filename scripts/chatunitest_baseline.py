"""ChatUniTest baseline wrapper (eval-plan E6).

ChatUniTest is an LLM-driven Java unit-test generator. Upstream:
<https://github.com/ChatUniTest/ChatUniTest>. Two integration modes:

  * **adapter** (default): we replicate ChatUniTest's prompt template against
    our own provider cache and parse the generated tests. This avoids upstream
    coupling and stays consistent with our cache invariants. Requires API access.

  * **subprocess**: shell out to a local checkout of ChatUniTest's CLI and use
    its native generation pipeline. Requires ``--chatunitest-root`` and that
    upstream's environment is set up.

This module is API-gated. In the absence of credentials, the script prints the
prompt that would be sent and writes a stub events.jsonl row so downstream
aggregation does not break.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagetransval.schema import load_subject


CHATUNITEST_PROMPT = """\
You are ChatUniTest, generating JUnit tests for the Java class below.
Produce 8-12 diverse test methods covering:
  * happy-path inputs,
  * boundary cases (empty, null, max/min, off-by-one),
  * documented exceptions,
  * branch coverage.

Output a single Java class named ``Solution_ChatUniTest`` containing only the
@Test methods. Do not include any prose.

Class under test:
```
{target_source}
```
Entry-point method: ``{entrypoint}``
"""


def build_prompt(subject_dir: pathlib.Path) -> tuple[str, str]:
    """Return (prompt, target_source) for the subject."""
    subject = load_subject(subject_dir / "metadata.json")
    if subject.target_language != "java" or subject.target_path is None:
        raise ValueError("ChatUniTest baseline is Java-only and requires translated target")
    target = pathlib.Path(subject.target_path).read_text()
    return CHATUNITEST_PROMPT.format(target_source=target, entrypoint=subject.entrypoint), target


def run_adapter(subject_dir: pathlib.Path, out_dir: pathlib.Path,
                provider: str, model: str) -> dict[str, Any]:
    subject = load_subject(subject_dir / "metadata.json")
    work = out_dir / subject.id
    work.mkdir(parents=True, exist_ok=True)
    try:
        prompt, target = build_prompt(subject_dir)
    except ValueError as e:
        return {"subject": subject.id, "skipped": True, "reason": str(e)}

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return {"subject": subject.id, "skipped": True, "reason": "OPENAI_API_KEY missing"}
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1500,
        )
        raw = resp.choices[0].message.content or ""
    elif provider == "gemini":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return {"subject": subject.id, "skipped": True, "reason": "GOOGLE_API_KEY missing"}
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        client = genai.GenerativeModel(model)
        raw = client.generate_content(prompt, generation_config={"temperature": 0, "max_output_tokens": 1500}).text
    else:
        return {"subject": subject.id, "skipped": True, "reason": f"unknown_provider:{provider}"}

    test_file = work / "Solution_ChatUniTest.java"
    test_file.write_text(raw)

    # Reuse the SBST EvoSuite-style Java argument extractor for test-input mining.
    from scripts.sbst_baseline import _extract_evosuite_inputs as extract_java_inputs
    # ChatUniTest tests use Solution.method(...) syntax, same as EvoSuite.
    # The extractor walks evosuite-tests/; here we point it at the file's parent.
    (work / "evosuite-tests").mkdir(exist_ok=True)
    (work / "evosuite-tests" / "Solution_ESTest.java").write_text(raw)
    inputs = extract_java_inputs(work, subject.entrypoint)
    return {
        "subject": subject.id,
        "mode": "chatunitest",
        "provider": provider,
        "model": model,
        "generated_inputs": inputs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adapter", "subprocess"], default="adapter")
    ap.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--subjects-root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "subprocess":
        print("subprocess mode not implemented yet; use --mode adapter", file=sys.stderr)
        return 2
    n = 0
    records = []
    for subject_dir in sorted(args.subjects_root.iterdir()):
        if not (subject_dir / "metadata.json").exists():
            continue
        rec = run_adapter(subject_dir, args.out, args.provider, args.model)
        records.append(rec)
        skipped = rec.get("skipped", False)
        print(f"  [chatunitest] {rec['subject']} {'SKIP:' + rec.get('reason', '') if skipped else 'OK inputs=' + str(len(rec.get('generated_inputs', [])))}")
        n += 1
        if args.limit and n >= args.limit:
            break
    summary = args.out / "chatunitest_summary.jsonl"
    with summary.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
