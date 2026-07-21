from __future__ import annotations

from pathlib import Path


def read_sectioned_lines(path: Path) -> list[str]:
    target = Path(path)
    if not target.exists():
        return []
    return target.read_text(encoding="utf-8").splitlines()


def write_sectioned_lines(path: Path, lines: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def section_content(lines: list[str], section_name: str) -> list[str]:
    header = f"#{section_name.strip().upper()}"
    start = None
    for idx, line in enumerate(lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].startswith("#"):
            end = idx
            break
    return list(lines[start:end])


def has_section(path: Path, section_name: str) -> bool:
    header = f"#{section_name.strip().upper()}"
    return any(line.strip().upper() == header for line in read_sectioned_lines(path))


def remove_section_from_lines(lines: list[str], section_name: str) -> list[str]:
    header = f"#{section_name.strip().upper()}"
    out: list[str] = []
    skip = False
    for line in lines:
        if line.strip().upper() == header:
            skip = True
            continue
        if skip:
            if line.startswith("#"):
                skip = False
                out.append(line)
            continue
        out.append(line)
    return out


def replace_section_in_lines(lines: list[str], section_name: str, content_lines: list[str]) -> list[str]:
    out = remove_section_from_lines(lines, section_name)
    while out and not out[-1].strip():
        out.pop()
    if out:
        out.append("")
    out.append(f"#{section_name.strip().upper()}")
    out.extend(content_lines)
    return out


def replace_section(path: Path, section_name: str, content_lines: list[str]) -> None:
    lines = read_sectioned_lines(path)
    write_sectioned_lines(path, replace_section_in_lines(lines, section_name, content_lines))
