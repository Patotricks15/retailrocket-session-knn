#!/usr/bin/env python3
"""Patch a literal white background rectangle into exported draw.io SVGs.

The ``rlespinasse/drawio-export`` SVG output sets
``style="background: #ffffff; ..."`` on the ``<svg>`` root, but GitHub's HTML
sanitizer strips inline ``style`` attributes from embedded SVGs, which makes
the diagram render with a transparent background (invisible text/lines in
dark mode). The fix that actually survives sanitization is a real
``<rect>`` element painted first, before any diagram content, so it is
read as opaque artwork rather than CSS.

This script finds each SVG's ``viewBox="0 0 W H"`` to size the rectangle and
inserts it as the first child of the outer ``<g>`` (the group that
immediately follows ``<defs/>``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
VIEWBOX_RE = re.compile(r'viewBox="0 0 (?P<w>[\d.]+) (?P<h>[\d.]+)"')
INSERTION_RE = re.compile(r"(<defs\s*/>\s*<g>)")


def patch_svg(path: Path) -> bool:
    """Insert the white background rect into one SVG file.

    Args:
        path: Path to an exported ``.svg`` file.

    Returns:
        ``True`` if the file was modified, ``False`` if it already carried
        the patch (idempotent -- safe to re-run).

    Raises:
        ValueError: If the file has no ``viewBox`` or no ``<defs/><g>``
            anchor to insert after.
    """
    text = path.read_text()
    if 'fill="#ffffff"/>' in text.split("<g>", 1)[-1][:80]:
        return False  # already patched

    viewbox_match = VIEWBOX_RE.search(text)
    if not viewbox_match:
        raise ValueError(f"{path} has no 'viewBox=\"0 0 W H\"' to size the background rect.")
    width, height = viewbox_match.group("w"), viewbox_match.group("h")

    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'

    anchor_match = INSERTION_RE.search(text)
    if not anchor_match:
        raise ValueError(f"{path} has no '<defs/><g>' anchor to insert the background rect after.")

    patched = text[: anchor_match.end()] + rect + text[anchor_match.end() :]
    path.write_text(patched)
    return True


def main() -> int:
    svgs = sorted(DOCS.glob("*.svg"))
    if not svgs:
        print(f"No SVGs found under {DOCS}; run the drawio-export step first.")
        return 1

    for svg in svgs:
        changed = patch_svg(svg)
        print(f"{'patched' if changed else 'already patched'} -> {svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
