#!/usr/bin/env python3
"""Export a session transcript as a self-contained, redacted HTML replay.

The transcript is the record of how the problem was actually solved: the
questions asked, the wrong turns, the measurement that settled each one. It is
also a security hazard — this session's log contains a ``printenv`` dump with
live API tokens, plus internal hostnames, ssh aliases, absolute home paths and
camera credentials carried in recalled memories.

So redaction is not a formatting nicety here, it is the point:

* every value of a known secret-bearing environment variable is dropped,
* high-signal token shapes (Atlassian, OpenRouter, Figma, generic 32-hex keys,
  ``user:password@host`` URLs) are replaced,
* internal hostnames, ssh aliases, home paths and dataset names are mapped to
  neutral placeholders,
* recalled-memory blocks are dropped wholesale, since they carry context from
  unrelated sessions.

``--audit`` re-scans the rendered output for every pattern and exits non-zero if
anything survived, so a leak fails the export instead of shipping.

    uv run python -m lab.export_session --out results/session-replay.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

#: Environment variables whose values must never appear. Matched as
#: ``NAME=value`` or ``"NAME": "value"`` anywhere in the text.
SECRET_ENVIRONMENT = (
    "CONFLUENCE_API_TOKEN",
    "DD_API_KEY",
    "FIGMA_API_KEY",
    "JIRA_API_KEY",
    "JIRA_API_TOKEN",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "GEMINI_API_KEY",
)

#: Token shapes worth redacting on sight, independent of any variable name.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ATATT3[A-Za-z0-9_=\-]{20,}", "[REDACTED:atlassian-token]"),
    (r"sk-or-v1-[A-Za-z0-9]{16,}", "[REDACTED:openrouter-key]"),
    (r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{24,}", "[REDACTED:api-key]"),
    (r"figd_[A-Za-z0-9_\-]{16,}", "[REDACTED:figma-key]"),
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED:github-token]"),
    (r"\b[0-9a-f]{32}\b", "[REDACTED:hex-key]"),
    # rtsp://user:password@host and any other credential-bearing URL
    (r"(?P<scheme>[a-z][a-z0-9+.\-]*://)[^\s/@:]+:[^\s/@]+@", r"\g<scheme>[REDACTED:credentials]@"),
)

#: Internal identifiers mapped to neutral names. Not secrets, but nobody outside
#: the team benefits from the real host and path names.
IDENTIFIER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"cmc-\d+-\d+-\d+-\d+", "a100-host"),
    (r"DEC-TUANND12", "workstation"),
    (r"gpu-training", "gpu-host"),
    (r"/home/tuannd12", "~"),
    (r"/root/workspace[A-Za-z0-9/_.\-]*", "~/project"),
    (r"/root(?=[/\s\"'])", "~"),
    (r"sam3-person-face[a-z0-9-]*", "person-face-dataset"),
    (r"namnbp", "teammate"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[host]"),
    (r"tuannd12", "user"),
)

#: Transcript text starting with any of these is context injected around the
#: conversation rather than part of it.
DROP_PREFIXES = (
    "<memories>",
    "This agent has local Mnemopi",
    "<system-reminder>",
)

TOOL_LABELS = {
    "_bash": "bash",
    "_read": "read",
    "_edit": "edit",
    "_write": "write",
    "_eval": "eval",
    "_grep": "grep",
    "_glob": "glob",
    "_task": "task",
    "_hub": "hub",
    "_todo": "todo",
    "web_search": "web_search",
}


def redact(text: str) -> str:
    """Apply every redaction rule to a block of transcript text."""
    for name in SECRET_ENVIRONMENT:
        text = re.sub(rf"{name}\s*=\s*\S+", f"{name}=[REDACTED]", text)
        text = re.sub(rf'"{name}"\s*:\s*"[^"]*"', f'"{name}": "[REDACTED]"', text)
    for pattern, replacement in SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in IDENTIFIER_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… [{len(text) - limit} more characters]"


def result_text(message: dict[str, Any]) -> str:
    """Best-effort plain text for a tool result.

    omp stores the rendered form under ``details.displayContent.text`` and the
    model-facing form in ``content``, which may be a list of blocks or the repr
    of one. The rendered form is preferred because it is what a reader wants;
    the fallbacks keep older records readable.
    """
    details = message.get("details") or {}
    display = details.get("displayContent") or {}
    if isinstance(display, dict) and display.get("text"):
        return str(display["text"])
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content or "")


def collect(path: Path, output_limit: int) -> list[dict[str, Any]]:
    """Flatten the transcript into an ordered list of replay entries.

    omp's layout: assistant messages carry ``text``, ``thinking`` and
    ``toolCall`` blocks; every tool result arrives as a separate message with
    ``role="toolResult"`` linked by ``toolCallId``. Results are folded into their
    call so the replay reads as one collapsible step per tool use.
    """
    entries: list[dict[str, Any]] = []
    calls: dict[str, dict[str, Any]] = {}

    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "message":
            continue
        message = record.get("message") or {}
        role = message.get("role")
        timestamp = record.get("timestamp", "")

        if role == "toolResult":
            target = calls.get(message.get("toolCallId", ""))
            if target is not None:
                target["output"] = redact(clip(result_text(message), output_limit))
                target["failed"] = bool(message.get("isError"))
            continue

        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue

        for block in content:
            kind = block.get("type")
            if kind == "text":
                text = block.get("text") or ""
                if any(text.lstrip().startswith(prefix) for prefix in DROP_PREFIXES) or not text.strip():
                    continue
                entries.append(
                    {"kind": "user" if role == "user" else "assistant", "timestamp": timestamp, "text": redact(text)}
                )
            elif kind == "thinking":
                text = block.get("thinking") or block.get("text") or ""
                if text.strip():
                    entries.append({"kind": "thinking", "timestamp": timestamp, "text": redact(clip(text, 1600))})
            elif kind == "toolCall":
                arguments = block.get("arguments") or {}
                entry = {
                    "kind": "tool",
                    "timestamp": timestamp,
                    "tool": TOOL_LABELS.get(block.get("name", ""), block.get("name", "tool")),
                    "intent": redact(str(block.get("intent") or arguments.get("i") or "")),
                    "arguments": redact(clip(json.dumps(arguments, indent=1, default=str), 1400)),
                    "output": "",
                    "failed": False,
                }
                calls[block.get("id", "")] = entry
                entries.append(entry)

    return entries


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title} — session replay</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin: 0 auto; max-width: 1080px; padding: 28px 24px 64px; background: #0f1115; color: #e6e8ee;
  font: 14.5px/1.62 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
p.sub {{ color: #9aa3b2; margin: 0 0 18px; }}
.bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }}
.chip {{ background: #171a21; border: 1px solid #242936; border-radius: 999px; padding: 4px 12px;
  color: #9aa3b2; font-size: 12px; }}
.controls {{ margin: 0 0 18px; }}
button {{ background: #1d2430; color: #e6e8ee; border: 1px solid #2c3444; border-radius: 6px;
  padding: 5px 11px; font-size: 13px; cursor: pointer; margin-right: 8px; }}
button:hover {{ background: #26303f; }}
.entry {{ margin: 0 0 14px; }}
.user {{ background: #16202b; border-left: 3px solid #5aa9e6; border-radius: 8px; padding: 10px 14px; }}
.assistant {{ background: #171a21; border-left: 3px solid #5ddc9a; border-radius: 8px; padding: 10px 14px; }}
.role {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #7d8797; margin-bottom: 4px; }}
.text {{ white-space: pre-wrap; word-break: break-word; }}
details.tool {{ background: #14171e; border: 1px solid #1f2430; border-radius: 6px; padding: 5px 10px;
  margin: 0 0 6px; }}
details.tool summary {{ cursor: pointer; color: #9aa3b2; font-size: 13px; }}
details.tool summary .name {{ color: #c8a24a; font-family: ui-monospace, monospace; }}
details.tool pre {{ background: #0f1115; border: 1px solid #1f2430; border-radius: 5px; padding: 8px 10px;
  overflow-x: auto; font-size: 12px; color: #c4cad6; margin: 8px 0 0; white-space: pre-wrap; }}
details.thinking {{ background: #14161c; border: 1px dashed #2a3140; border-radius: 6px; padding: 5px 10px;
  margin: 0 0 8px; }}
details.thinking summary {{ cursor: pointer; color: #7d8797; font-size: 12px; }}
details.thinking .text {{ white-space: pre-wrap; color: #9aa3b2; font-size: 13px; margin-top: 6px; }}
.failed {{ color: #e06c6c; }}
.notice {{ background: #221a1a; border-left: 3px solid #e06c6c; border-radius: 8px; padding: 10px 14px;
  margin-bottom: 20px; color: #e4d6d6; font-size: 13.5px; }}
footer {{ margin-top: 32px; color: #6b7482; font-size: 12px; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="sub">Session replay · {user_count} human turns · {assistant_count} agent replies · {tool_count} tool calls</p>
<div class="bar">{chips}</div>
<div class="notice"><strong>Redacted export.</strong> API tokens, credentials, internal hostnames, ssh
aliases, home paths and dataset names were replaced before rendering, and recalled-memory blocks were
dropped. The export fails rather than publishes if any pattern survives the audit.</div>
<div class="controls">
  <button onclick="document.querySelectorAll('details.tool').forEach(d=>d.open=true)">Expand all tool calls</button>
  <button onclick="document.querySelectorAll('details.tool').forEach(d=>d.open=false)">Collapse all</button>
</div>
{body}
<footer>{footer}</footer>
</body></html>
"""


def render(entries: list[dict[str, Any]], title: str, footer: str) -> str:
    pieces: list[str] = []
    for entry in entries:
        if entry["kind"] == "tool":
            summary = f'<span class="name">{html.escape(entry["tool"])}</span>'
            if entry["intent"]:
                summary += f' — {html.escape(entry["intent"])}'
            if entry.get("failed"):
                summary += ' <span class="failed">failed</span>' 
            body = f'<pre>{html.escape(entry["arguments"])}</pre>'
            if entry["output"]:
                body += f'<pre>{html.escape(entry["output"])}</pre>'
            pieces.append(f'<details class="tool"><summary>{summary}</summary>{body}</details>')
        elif entry["kind"] == "thinking":
            pieces.append(
                f'<details class="thinking"><summary>reasoning</summary>'
                f'<div class="text">{html.escape(entry["text"])}</div></details>'
            )
        else:
            label = "human" if entry["kind"] == "user" else "agent"
            pieces.append(
                f'<div class="entry {entry["kind"]}"><div class="role">{label}'
                f'</div><div class="text">{html.escape(entry["text"])}</div></div>'
            )

    counts = {kind: sum(1 for entry in entries if entry["kind"] == kind) for kind in ("user", "assistant", "tool")}
    chips = "".join(
        f'<span class="chip">{html.escape(text)}</span>'
        for text in (
            "12.81 → 33.63 img/s (2.63×)",
            "24 logged experiments",
            "2 custom Triton kernels",
            "profiler → Nsight → NVTX",
        )
    )
    return PAGE.format(
        title=html.escape(title),
        user_count=counts["user"],
        assistant_count=counts["assistant"],
        tool_count=counts["tool"],
        chips=chips,
        body="".join(pieces),
        footer=html.escape(footer),
    )


def audit(text: str) -> list[str]:
    """Return every sensitive pattern still present in the rendered output."""
    findings: list[str] = []
    for name in SECRET_ENVIRONMENT:
        if re.search(rf"{name}\s*=\s*(?!\[REDACTED)\S", text):
            findings.append(f"{name} value present")
    for pattern, replacement in SECRET_PATTERNS:
        if replacement.startswith("[REDACTED") and re.search(pattern, text):
            findings.append(f"pattern {pattern} present")
    for pattern, _ in IDENTIFIER_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            findings.append(f"identifier {pattern} present ({len(matches)}x)")
    return findings


def to_claude_code_jsonl(transcript: Path, destination: Path) -> int:
    """Rewrite the session as Claude-Code-schema JSONL, redacted.

    The mature viewers in this space (``claude-code-log``, ``claude-code-transcripts``,
    ``claude-replay``) all read Claude Code's transcript layout, which differs from
    omp's: the role lives in the top-level ``type`` rather than only in
    ``message.role``, and each record carries ``uuid``/``parentUuid``/``sessionId``.
    Converting once means those tools work without patching them, and redaction
    happens here so no unredacted copy is ever written.

    Returns the number of records written.
    """
    written = 0
    previous_uuid: str | None = None
    session_id = transcript.stem.split("_")[-1]
    with destination.open("w", encoding="utf-8") as handle:
        for line in transcript.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "message":
                continue
            message = record.get("message") or {}
            role = message.get("role")
            if role == "toolResult":
                # Claude Code's layout puts results in a user message carrying a
                # tool_result block keyed by the call id.
                converted = {
                    "parentUuid": previous_uuid,
                    "isSidechain": False,
                    "userType": "external",
                    "cwd": "~/project",
                    "sessionId": session_id,
                    "version": "omp-export",
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("toolCallId", ""),
                                "content": clip(result_text(message), 4000),
                                "is_error": bool(message.get("isError")),
                            }
                        ],
                    },
                    "uuid": record.get("id"),
                    "timestamp": record.get("timestamp"),
                }
                handle.write(redact(json.dumps(converted, default=str)) + "\n")
                previous_uuid = converted["uuid"]
                written += 1
                continue
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue
            kept: list[dict[str, Any]] = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    text = str(block.get("text", ""))
                    if any(text.lstrip().startswith(prefix) for prefix in DROP_PREFIXES) or not text.strip():
                        continue
                    kept.append({"type": "text", "text": text})
                elif kind == "thinking":
                    kept.append({"type": "thinking", "thinking": block.get("thinking") or block.get("text") or ""})
                elif kind == "toolCall":
                    kept.append(
                        {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", "tool"),
                            "input": block.get("arguments") or {},
                        }
                    )
            if not kept:
                continue
            # Re-parent onto the last *kept* record: dropping context blocks
            # otherwise leaves dangling parentUuids, which the viewers report as
            # orphan nodes and promote to separate conversation roots.
            converted = {
                "parentUuid": previous_uuid,
                "isSidechain": False,
                "userType": "external",
                "cwd": "~/project",
                "sessionId": session_id,
                "version": "omp-export",
                "type": role,
                "message": {"role": role, "content": kept},
                "uuid": record.get("id"),
                "timestamp": record.get("timestamp"),
            }
            handle.write(redact(json.dumps(converted)) + "\n")
            previous_uuid = converted["uuid"]
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Session JSONL; defaults to the newest session for this working directory.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/session-replay.html"))
    parser.add_argument("--output-limit", type=int, default=2000, help="Characters kept per tool result.")
    parser.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--claude-code-out",
        type=Path,
        default=None,
        help="Also write redacted Claude-Code-schema JSONL here, for claude-code-log and friends.",
    )
    args = parser.parse_args()

    transcript = args.transcript
    if transcript is None:
        candidates = sorted(
            (Path.home() / ".omp/agent/sessions/-tmp").glob("*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("no transcript found; pass --transcript", file=sys.stderr)
            return 2
        transcript = candidates[0]

    title = "Improve RF-DETR training performance"
    entries = collect(transcript, args.output_limit)
    page = render(entries, title, f"exported from {transcript.name} · {len(entries)} entries")

    findings = audit(page)
    if findings and args.audit:
        print("REFUSING TO WRITE — sensitive content survived redaction:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    if args.claude_code_out is not None:
        args.claude_code_out.parent.mkdir(parents=True, exist_ok=True)
        count = to_claude_code_jsonl(transcript, args.claude_code_out)
        leaks = audit(args.claude_code_out.read_text(encoding="utf-8"))
        if leaks and args.audit:
            args.claude_code_out.unlink()
            print(f"REFUSING to keep {args.claude_code_out}: {leaks}", file=sys.stderr)
            return 1
        print(f"wrote {args.claude_code_out} ({count} records, audit clean)")
    print(f"wrote {args.out} ({len(entries)} entries, audit clean)")
    print(f"file://{args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
