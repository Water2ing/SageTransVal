"""Cache-key generation that incorporates harness and schema versions.

Existing caches are keyed by (provider, model, prompt, decoding). Extensions
that change the harness or the subject schema would silently reuse stale
translations or stale validation outputs unless the cache key invalidates.
This module centralizes the cache key construction so every callsite uses
the same recipe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .schema import HARNESS_VERSION, SCHEMA_VERSION


def make_translation_cache_key(*, provider: str, model: str, prompt: str,
                               decoding: Mapping[str, Any],
                               schema_version: int = SCHEMA_VERSION,
                               harness_version: int = HARNESS_VERSION) -> str:
    payload = {
        "kind": "translation",
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "decoding": dict(decoding),
        "schema_version": schema_version,
        "harness_version": harness_version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_validation_cache_key(*, subject_id: str, mode: str, provider: str,
                              model: str,
                              schema_version: int = SCHEMA_VERSION,
                              harness_version: int = HARNESS_VERSION) -> str:
    payload = {
        "kind": "validation",
        "subject": subject_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "schema_version": schema_version,
        "harness_version": harness_version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
