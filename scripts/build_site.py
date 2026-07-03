from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"
SITE_DIR = ROOT / "site"
CHAPTERS_DIR = SITE_DIR / "chapters"
REPO_TUTORIALS_URL = "https://github.com/fengjikui/llm-tutor/blob/main/tutorials"


@dataclass(frozen=True)
class Chapter:
    number: int
    stem: str
    source_path: Path
    output_path: Path
    title: str
    display_title: str
    summary: str
    status: str
    phase: str
    phase_label: str
    tags: list[str]
    body: str

    @property
    def url(self) -> str:
        return f"./chapters/{self.stem}.html"

    @property
    def github_url(self) -> str:
        return f"{REPO_TUTORIALS_URL}/{self.source_path.name}"


PHASES = {
    "foundation": "最小地基",
    "sequence": "视觉与序列",
    "transformer": "Transformer",
    "post-training": "Post-training",
    "modern": "现代扩展",
}


TAGS_BY_NUMBER = {
    1: ["ML", "分类", "训练循环"],
    2: ["loss", "gradient", "SGD"],
    3: ["PyTorch", "DataLoader", "loop"],
    4: ["NN", "ReLU", "表示学习"],
    5: ["optimizer", "dropout", "schedule"],
    6: ["CNN", "vision", "Fashion-MNIST"],
    7: ["RNN", "sequence", "BPTT"],
    8: ["LSTM", "GRU", "gates"],
    9: ["Seq2Seq", "translation", "decoder"],
    10: ["attention", "alignment", "context"],
    11: ["self-attention", "mask", "QKV"],
    12: ["Transformer", "LayerNorm", "MLP"],
    13: ["GPT", "decoder-only", "LM head"],
    14: ["next-token", "loss", "causal"],
    15: ["pretrain", "checkpoint", "sampling"],
    16: ["SFT", "instruction", "mask"],
    17: ["PPO", "RLHF", "policy"],
    18: ["DPO", "preference", "KL"],
    19: ["GRPO", "RLVR", "group"],
    20: ["capstone", "pipeline", "LLM"],
    21: ["RoPE", "GQA", "MLA"],
    22: ["ClipCap", "Qwen3-VL", "VLA"],
    23: ["DDPM", "Stable Diffusion", "DiT"],
}


SPECIAL_LINE_RE = re.compile(
    r"^(```|#{1,6}\s+|[-*]\s+|\d+\.\s+|>\s*|\|.*\|)"
)
INLINE_RE = re.compile(r"(`[^`]+`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*)")


def main() -> None:
    chapters = load_chapters()
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    for old_page in CHAPTERS_DIR.glob("*.html"):
        old_page.unlink()

    for index, chapter in enumerate(chapters):
        previous = chapters[index - 1] if index > 0 else None
        next_chapter = chapters[index + 1] if index + 1 < len(chapters) else None
        chapter.output_path.write_text(
            render_chapter_page(chapter, previous, next_chapter),
            encoding="utf-8",
        )

    data = [chapter_to_data(chapter) for chapter in chapters]
    (SITE_DIR / "chapters-data.js").write_text(
        "window.LLM_TUTOR_CHAPTERS = "
        + json.dumps(data, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    (SITE_DIR / "chapters.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(chapters)} chapters into {CHAPTERS_DIR}")


def load_chapters() -> list[Chapter]:
    chapters: list[Chapter] = []
    for path in sorted(TUTORIALS_DIR.glob("[0-9][0-9]_*.md")):
        number_match = re.match(r"^(\d{2})_", path.name)
        if number_match is None:
            continue
        number = int(number_match.group(1))
        metadata, body = split_frontmatter(path.read_text(encoding="utf-8"))
        title = str(metadata.get("title") or first_heading(body) or path.stem)
        phase = phase_for_number(number)
        chapters.append(
            Chapter(
                number=number,
                stem=path.stem,
                source_path=path,
                output_path=CHAPTERS_DIR / f"{path.stem}.html",
                title=title,
                display_title=strip_number_prefix(title),
                summary=str(metadata.get("summary") or ""),
                status=str(metadata.get("status") or ""),
                phase=phase,
                phase_label=PHASES[phase],
                tags=TAGS_BY_NUMBER.get(number, []),
                body=body.strip(),
            )
        )
    return chapters


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, text[end + 5 :]


def first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def strip_number_prefix(title: str) -> str:
    return re.sub(r"^\d{1,2}\.\s*", "", title).strip()


def phase_for_number(number: int) -> str:
    if number <= 5:
        return "foundation"
    if number <= 10:
        return "sequence"
    if number <= 15:
        return "transformer"
    if number <= 20:
        return "post-training"
    return "modern"


def chapter_to_data(chapter: Chapter) -> dict[str, Any]:
    return {
        "no": f"{chapter.number:02d}",
        "title": chapter.display_title,
        "summary": chapter.summary,
        "phase": chapter.phase,
        "phaseLabel": chapter.phase_label,
        "tags": chapter.tags,
        "file": chapter.source_path.name,
        "url": chapter.url,
        "githubUrl": chapter.github_url,
    }


def render_chapter_page(
    chapter: Chapter,
    previous: Chapter | None,
    next_chapter: Chapter | None,
) -> str:
    rendered = render_markdown(chapter.body)
    toc = table_of_contents(chapter.body)
    previous_link = (
        nav_link(previous, "上一篇")
        if previous
        else '<span class="article-nav-empty">上一篇</span>'
    )
    next_link = (
        nav_link(next_chapter, "下一篇")
        if next_chapter
        else '<span class="article-nav-empty">下一篇</span>'
    )
    toc_html = "\n".join(
        f'<a class="toc-level-{level}" href="#{heading_id}">{escape_text(text)}</a>'
        for level, heading_id, text in toc
    )
    tags_html = "".join(f"<span>{escape_text(tag)}</span>" for tag in chapter.tags)
    title = escape_text(chapter.title)
    summary = escape_text(chapter.summary)
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="{summary}" />
    <title>{title} | Jikui Notes</title>
    <link rel="stylesheet" href="../styles.css" />
  </head>
  <body>
    <header class="site-header">
      <a class="brand" href="../index.html#top" aria-label="Jikui Notes">
        <span class="brand-mark" aria-hidden="true">J</span>
        <span>Jikui Notes</span>
      </a>
      <nav class="top-nav" aria-label="主导航">
        <a href="../index.html#route">路线</a>
        <a href="../index.html#chapters">章节</a>
        <a href="../index.html#experiments">实验</a>
        <a href="../index.html#references">参考</a>
      </nav>
      <a class="header-action" href="{chapter.github_url}">GitHub</a>
    </header>

    <main class="article-main">
      <section class="article-hero">
        <div>
          <a class="article-back" href="../index.html#chapters">返回章节列表</a>
          <p class="series-label">
            {escape_text(chapter.phase_label)} / Chapter {chapter.number:02d}
          </p>
          <h1>{title}</h1>
          <p class="article-summary">{summary}</p>
          <div class="article-tags">{tags_html}</div>
        </div>
      </section>

      <section class="article-shell">
        <aside class="article-toc" aria-label="本文目录">
          <strong>本文目录</strong>
          <nav>{toc_html}</nav>
        </aside>
        <article class="markdown-body">
          {rendered}
        </article>
      </section>

      <nav class="article-nav" aria-label="章节导航">
        {previous_link}
        {next_link}
      </nav>
    </main>

    <footer class="site-footer">
      <span>Jikui Notes · 从梯度下降到 GPT</span>
      <a href="{chapter.github_url}">在 GitHub 查看本章</a>
    </footer>
  </body>
</html>
"""


def nav_link(chapter: Chapter, label: str) -> str:
    return (
        f'<a href="./{chapter.stem}.html">'
        f"<span>{label}</span><strong>{escape_text(chapter.display_title)}</strong>"
        "</a>"
    )


def table_of_contents(markdown: str) -> list[tuple[int, str, str]]:
    headings: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}
    for line in markdown.splitlines():
        match = re.match(r"^(#{2,3})\s+(.+)$", line)
        if not match:
            continue
        level = len(match.group(1))
        text = strip_markdown(match.group(2).strip())
        heading_id = unique_heading_id(text, seen)
        headings.append((level, heading_id, text))
    return headings


def render_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    heading_ids: dict[str, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            html_block, index = render_code_block(lines, index)
            output.append(html_block)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            if level == 1:
                index += 1
                continue
            text = strip_markdown(heading.group(2).strip())
            heading_id = unique_heading_id(text, heading_ids)
            output.append(
                f'<h{level} id="{heading_id}">{render_inline(heading.group(2).strip())}</h{level}>'
            )
            index += 1
            continue

        if is_table_start(lines, index):
            html_block, index = render_table(lines, index)
            output.append(html_block)
            continue

        if re.match(r"^[-*]\s+", stripped):
            html_block, index = render_list(lines, index, ordered=False)
            output.append(html_block)
            continue

        if re.match(r"^\d+\.\s+", stripped):
            html_block, index = render_list(lines, index, ordered=True)
            output.append(html_block)
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            output.append(f"<blockquote>{render_inline(' '.join(quote_lines))}</blockquote>")
            continue

        paragraph: list[str] = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or SPECIAL_LINE_RE.match(candidate):
                break
            paragraph.append(candidate)
            index += 1
        output.append(f"<p>{render_inline(' '.join(paragraph))}</p>")

    return "\n".join(output)


def render_code_block(lines: list[str], index: int) -> tuple[str, int]:
    opening = lines[index].strip()
    language = opening.removeprefix("```").strip()
    index += 1
    code_lines: list[str] = []
    while index < len(lines) and not lines[index].strip().startswith("```"):
        code_lines.append(lines[index])
        index += 1
    if index < len(lines):
        index += 1
    language_label = escape_text(language) if language else "code"
    class_name = f' class="language-{escape_text(language)}"' if language else ""
    return (
        '<figure class="code-block">'
        f"<figcaption>{language_label}</figcaption>"
        f"<pre><code{class_name}>{escape_text(chr(10).join(code_lines))}</code></pre>"
        "</figure>",
        index,
    )


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index].strip()
    separator = lines[index + 1].strip()
    return "|" in current and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", separator)


def render_table(lines: list[str], index: int) -> tuple[str, int]:
    raw_lines: list[str] = []
    while index < len(lines) and "|" in lines[index]:
        raw_lines.append(lines[index].strip())
        index += 1
    header = split_table_row(raw_lines[0])
    rows = [split_table_row(row) for row in raw_lines[2:]]
    header_html = "".join(f"<th>{render_inline(cell)}</th>" for cell in header)
    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header_html
        + "</tr></thead><tbody>"
        + rows_html
        + "</tbody></table></div>",
        index,
    )


def split_table_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def render_list(lines: list[str], index: int, *, ordered: bool) -> tuple[str, int]:
    items: list[str] = []
    pattern = re.compile(r"^\d+\.\s+(.+)$" if ordered else r"^[-*]\s+(.+)$")
    while index < len(lines):
        match = pattern.match(lines[index].strip())
        if not match:
            break
        items.append(f"<li>{render_inline(match.group(1).strip())}</li>")
        index += 1
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(items) + f"</{tag}>", index


def render_inline(text: str) -> str:
    output: list[str] = []
    position = 0
    for match in INLINE_RE.finditer(text):
        output.append(escape_text(text[position : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            output.append(f"<code>{escape_text(token[1:-1])}</code>")
        elif token.startswith("["):
            label = escape_text(match.group(2) or "")
            href = normalize_href(match.group(3) or "")
            output.append(f'<a href="{escape_attr(href)}">{label}</a>')
        elif token.startswith("**"):
            output.append(f"<strong>{escape_text(match.group(4) or '')}</strong>")
        position = match.end()
    output.append(escape_text(text[position:]))
    return "".join(output)


def normalize_href(href: str) -> str:
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    target = Path(href)
    if target.suffix == ".md" and re.match(r"^\d{2}_", target.name):
        return f"./{target.stem}.html"
    if target.suffix == ".md":
        return f"https://github.com/fengjikui/llm-tutor/blob/main/{href.lstrip('./')}"
    return href


def strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "")
    return text


def unique_heading_id(text: str, seen: dict[str, int]) -> str:
    base = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.lower()).strip("-")
    if not base:
        base = "section"
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count + 1}"


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def escape_attr(text: str) -> str:
    return html.escape(text, quote=True)


if __name__ == "__main__":
    main()
