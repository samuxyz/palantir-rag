import anthropic
from app.config import settings
from app.models import Source

_client: anthropic.Anthropic | None = None


def get_claude_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


# ── Prompt template ──────────────────────────────────────────────────────────
# This is v1. We'll iterate on it together.
#
# Design choices to discuss:
#   1. The persona: "knowledgeable guide" vs "strict extractor" vs "narrator"
#   2. The grounding rule: "ONLY the passages" vs "passages + your knowledge"
#   3. How we format each source passage (numbered? with location? with score?)
#   4. Where the question goes: after context (current) vs before
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a lore-keeper of Middle-earth — deeply versed in its history, peoples, and ages. \
You speak with the warmth and gravity of a storyteller who lived through these events, \
not as a scholar cataloguing facts.

Answer using ONLY the passages provided. Do not draw on outside knowledge.
If the passages do not contain the answer, say so plainly — even a lore-keeper has limits.

Guidelines for your voice:
- Write in flowing prose, not bullet points or headers. Let the answer breathe.
- Favour narrative over enumeration: "He wandered long in the wild before..." rather than "Key facts: 1. He wandered..."
- Reference sources with inline citations like [1] naturally woven into the prose, not bolted on at the end.
- Keep the register warm but authoritative — like a trusted elder recounting a tale by firelight.
- Do not begin your answer with "Based on the passages provided" or any similar preamble. Just speak.\
"""


def _format_sources(sources: list[Source]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[{i}] {s.chapter_or_page}\n{s.text}")
    return "\n\n".join(parts)


def generate(query: str, sources: list[Source]) -> str:
    """Call Claude to answer query, grounded in the retrieved sources."""
    context = _format_sources(sources)
    user_message = f"Passages:\n\n{context}\n\nQuestion: {query}"

    client = get_claude_client()
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = stream.get_final_message()

    for block in message.content:
        if block.type == "text":
            return block.text
    return ""
