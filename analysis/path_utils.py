"""
Utilities for extracting plausible repository-relative file paths from free text.

This is used to augment hypothesis source context by guessing additional files
mentioned in strategist outputs, hypothesis descriptions, or evidence notes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

_EXTENSIONS = (
    "rs", "sol", "vy", "json", "toml", "yaml", "yml", "py", "go",
    "ts", "js", "tsx", "jsx", "c", "h", "cpp", "hpp", "java", "kt",
    "swift", "rb", "php", "cs", "scala", "sql", "sh", "md", "txt",
    "ini", "cfg"
)

# Require at least one path separator for most matches to reduce false positives.
# Allow top-level config files like Cargo.toml or package.json as a special case
# when they actually exist in the repository root.
_PATH_RE = re.compile(
    rf"(?P<raw>(?:[A-Za-z0-9_./\\-]{{1,200}})\.(?:{'|'.join(_EXTENSIONS)}))"
)


def _clean_candidate(p: str) -> str:
    """Normalize and strip punctuation around a candidate path-like string."""
    p = p.strip().strip("`'\"()[]{}<>")
    # Normalize backslashes from logs into POSIX separators
    p = p.replace("\\", "/")
    # Collapse duplicate slashes
    while "//" in p:
        p = p.replace("//", "/")
    return p


def guess_relpaths(text: str | None, repo_root: Path | None = None, *, extra_texts: Iterable[str] | None = None, max_paths: int = 20) -> list[str]:
    """Extract plausible repo-relative file paths from text.

    - Deduplicates results and returns only those that exist on disk when a repo_root is provided.
    - Prefers paths that include at least one subdirectory (e.g., "src/acl.rs").
    - Allows top-level files (e.g., "Cargo.toml") only if present in repo_root.
    """
    if not text and not extra_texts:
        return []

    corpus = []
    if text:
        corpus.append(text)
    if extra_texts:
        corpus.extend([t for t in extra_texts if isinstance(t, str) and t])

    found: list[str] = []
    seen: set[str] = set()

    for blob in corpus:
        for m in _PATH_RE.finditer(blob):
            raw = _clean_candidate(m.group("raw"))
            if not raw:
                continue
            # Basic sanity: ignore obviously non-path domains like "http://" or "https://"
            if raw.startswith("http://") or raw.startswith("https://"):
                continue
            # Require either a subdirectory or accept top-level only if it exists later
            has_sep = "/" in raw
            candidate = raw.lstrip("./")  # Treat as repo-relative

            # Skip very long segments or segments with spaces
            if len(candidate) > 240 or " " in candidate:
                continue

            # Optionally filter by existence
            if repo_root is not None:
                path = (repo_root / candidate)
                if has_sep:
                    if not path.exists():
                        continue
                else:
                    # Allow top-level files when they exist
                    if not path.exists():
                        continue

            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
                if len(found) >= max_paths:
                    return found

    return found


__all__ = ["guess_relpaths"]

