import json
from app.generation import get_claude_client

_SYSTEM = """\
Generate 4 different phrasings of the given question that would retrieve \
different relevant passages from a Lord of the Rings text corpus. \
Each phrasing should target a different angle or aspect of the question.
Respond with a JSON array of 4 strings and nothing else.\
"""


def expand_query(query: str) -> list[str]:
    """
    Generate alternative phrasings of a query to improve retrieval coverage.

    Why this helps: a single query embeds near one region of the vector space.
    Different phrasings embed near different regions, surfacing chunks that the
    original query would miss. Results from all phrasings are merged with RRF
    before reranking, so noise from weaker phrasings is naturally damped.
    """
    client = get_claude_client()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {query}"}],
    )
    try:
        expansions = json.loads(response.content[0].text)
        if isinstance(expansions, list):
            return [query] + [e for e in expansions if isinstance(e, str)]
    except (json.JSONDecodeError, IndexError):
        pass
    return [query]
