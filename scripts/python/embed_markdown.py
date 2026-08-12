"""
Inline a markdown report's local images and videos into ONE standalone HTML file.

    python scripts/python/embed_markdown.py <report.md> [--out report.html] [--tasks a,b]

Why this exists
---------------
The debug reports point at frames and videos on ``/project2`` through symlinks in the report
directory. That renders locally, but the markdown is not portable: move the .md anywhere and
every image is a dead link. This walks the same links, base64s the bytes into ``data:`` URIs,
and emits a single file that renders anywhere with no filesystem and no network.

Videos become ``<video controls preload="none">`` so a browser does not fetch every clip
before showing the first heading; images get ``loading="lazy"`` for the same reason. A full
50-episode report is ~13 MB, which is fine as a download and slow as a web page — pass
``--tasks`` to cut it to the episodes worth looking at.

Targets that are not local files (http, mailto, anchors) are left exactly as they are.
"""

import argparse
import base64
import mimetypes
import os
import re
import sys

import markdown

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
#: "## <task>" starts an episode section; used only by --tasks.
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --code:#f6f6f6; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#141414; --fg:#e8e8e8; --muted:#9a9a9a;
                                    --line:#2e2e2e; --code:#1d1d1d; }
}
html { background: var(--bg); }
body { background: var(--bg); color: var(--fg); margin: 0 auto; padding: 2rem 1.5rem 6rem;
       max-width: 60rem; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, sans-serif; }
h1 { font-size: 1.7rem; border-bottom: 2px solid var(--line); padding-bottom: .4rem; }
h2 { font-size: 1.3rem; margin-top: 2.5rem; border-bottom: 1px solid var(--line);
     padding-bottom: .3rem; }
h3 { font-size: 1.05rem; margin-top: 1.8rem; color: var(--muted); }
img { image-rendering: pixelated; max-width: 100%; border: 1px solid var(--line);
      border-radius: 3px; }
video { max-width: 100%; border: 1px solid var(--line); border-radius: 3px; }
pre { background: var(--code); padding: .8rem 1rem; border-radius: 4px; overflow-x: auto;
      font-size: 13px; }
code { background: var(--code); padding: .1rem .3rem; border-radius: 3px; font-size: 13px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: .35rem .6rem; text-align: left; }
details { margin: .5rem 0; }
summary { cursor: pointer; color: var(--muted); }
figcaption { color: var(--muted); font-size: 13px; }
blockquote { border-left: 3px solid var(--line); margin-left: 0; padding-left: 1rem;
             color: var(--muted); }
"""


def data_uri(path: str) -> str:
    """``data:<mime>;base64,...`` for one local file."""
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        return f"data:{mime};base64,{base64.b64encode(handle.read()).decode('ascii')}"


def select_tasks(text: str, wanted: list) -> str:
    """Keep the preamble plus only the ``## <task>`` sections matching *wanted* (substring)."""
    marks = list(HEADING_RE.finditer(text))
    if not marks:
        return text
    kept = [text[: marks[0].start()]]
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        if any(w.lower() in mark.group(1).lower() for w in wanted):
            kept.append(text[mark.start():end])
    return "".join(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="Path to the markdown report.")
    parser.add_argument("--out", default=None,
                        help="Output HTML path. Defaults to the report with a .html suffix.")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated substrings; keep only matching '## ' sections.")
    args = parser.parse_args()

    if not os.path.exists(args.report):
        print(f"no such report: {args.report}", file=sys.stderr)
        return 1
    base_dir = os.path.dirname(os.path.abspath(args.report))
    out_path = args.out or os.path.splitext(args.report)[0] + ".html"

    text = open(args.report).read()
    if args.tasks:
        text = select_tasks(text, [t.strip() for t in args.tasks.split(",") if t.strip()])

    # Cache keyed on the resolved path: the same frame is often linked twice, and base64ing a
    # multi-megabyte corpus twice over is pure waste.
    cache: dict = {}
    stats = {"img": 0, "video": 0, "missing": 0, "skipped": 0}

    def resolve(target: str):
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            stats["skipped"] += 1
            return None
        full = os.path.realpath(os.path.join(base_dir, target))
        if not os.path.isfile(full):
            stats["missing"] += 1
            return None
        if full not in cache:
            cache[full] = data_uri(full)
        return cache[full]

    def sub_image(match):
        alt, target = match.group(1), match.group(2)
        uri = resolve(target)
        if uri is None:
            return match.group(0)
        stats["img"] += 1
        return f'<img loading="lazy" alt="{alt}" src="{uri}">'

    def sub_link(match):
        label, target = match.group(1), match.group(2)
        if not target.lower().endswith((".mp4", ".webm", ".gif")):
            return match.group(0)
        uri = resolve(target)
        if uri is None:
            return match.group(0)
        stats["video"] += 1
        return (f'<figure><figcaption>{label}</figcaption>'
                f'<video controls preload="none" src="{uri}"></video></figure>')

    text = IMAGE_RE.sub(sub_image, text)
    text = LINK_RE.sub(sub_link, text)

    body = markdown.markdown(
        text,
        # md_in_html so the fenced code inside the report's <details> blocks still renders as
        # code rather than being dumped as literal backticks.
        extensions=["fenced_code", "tables", "md_in_html", "sane_lists"],
    )
    title = os.path.basename(os.path.splitext(out_path)[0])
    html = (f"<!doctype html>\n<html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>\n{body}\n</body></html>\n")

    with open(out_path, "w") as handle:
        handle.write(html)

    size = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}  ({size:.1f} MB)")
    print(f"  embedded {stats['img']} image(s), {stats['video']} video(s); "
          f"{stats['missing']} unresolved, {stats['skipped']} external left alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
