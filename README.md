# SageTransVal

Code and reproduction scripts for "Source-Obligation-Guided Validation of LLM-Based Code Translation".

## Repository contents

- `sagetransval/`: translation validation framework.
- `testgenrl/`: shared model, cache, and test-generation utilities.
- `scripts/`: experiment, audit, import, aggregation, and statistics scripts.
- `configs/` and `tests/`: experiment configurations and regression tests.
- `pyproject.toml`, `requirements.txt`, `Dockerfile`, and `package.json`: environment and dependency definitions.

This is a code-only repository. Input datasets and translation snapshots, generated run outputs and logs, reproduction outputs, and staged paper tables are intentionally excluded. The audit and aggregation commands therefore need the corresponding research inputs and archived logs supplied separately; this repository alone cannot reproduce the staged evidence or verify manuscript rates.

## Setup

Python 3.10 or newer is required. Install the research dependencies from the repository root:

```bash
python -m pip install -e ".[research]"
```

The C-to-Rust experiment also requires `gcc` or `clang` and `rustc`. Live provider runs require the relevant API keys in environment variables; do not put credentials in source files.

## Reproduction commands

Audit staged evidence after supplying the expected datasets and logs:

```bash
python scripts/reproduce_paper_pipeline.py audit
```

Run C-to-Rust v2 with live providers:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python scripts/reproduce_paper_pipeline.py run-crust-v2
```

Or replay provider translations from two snapshot directories. Each must contain a `manifest.json` and all 30 v2 Rust targets:

```bash
python scripts/reproduce_paper_pipeline.py run-crust-v2 \
  --openai-translations /path/to/openai/translation/root \
  --gemini-translations /path/to/gemini/translation/root
```

The runner validates the five main modes with the primary UB-safe policy (300 events total) and guided mode with that policy disabled (60 events), then writes a summary and repair packets under `reproduction/`. It refuses to overwrite an existing output directory. The required C-to-Rust source subjects must be supplied at `data/c_rust_v2_run/subjects/`; no datasets or run outputs are included here.

Python-to-Java statistical rechecks and repair revalidation likewise require the original translation snapshots and validation logs. The archived manuscript rates and staged tables are not part of this repository and are not modified by these scripts.
