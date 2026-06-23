from app.generation import get_claude_client

_SYSTEM = """\
Write a short passage (2-3 sentences) that directly answers the question, \
written as if it were an excerpt from a book or encyclopedia about Tolkien's Middle-earth. \
Be specific and factual. Do not add caveats or say you are generating a hypothetical.\
"""


def generate_hypothetical(query: str) -> str:
    """
    Generate a hypothetical answer to embed for HyDE retrieval.

    Why this helps: the raw query ('what color did Gandalf come back as?') embeds
    near other questions, not near passages that contain the answer. A hypothetical
    answer ('Gandalf returned as Gandalf the White...') embeds near the actual
    passage, so vector search finds the right chunk.
    """
    client = get_claude_client()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=150,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {query}"}],
    )
    return response.content[0].text
