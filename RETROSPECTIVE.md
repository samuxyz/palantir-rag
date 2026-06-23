# LOTR RAG — Project Retrospective

## What We Built

A production-quality Retrieval-Augmented Generation (RAG) system from scratch, in Python, with no RAG framework. Every component is visible and explicit:

```
epub/wiki → parse → chunk → embed → Chroma          (index phase)

query → embed → vector search ─┐
      → BM25 search            ├→ RRF merge → rerank → Claude → answer
                               ┘
```

Built with: FastAPI, ChromaDB, sentence-transformers, rank-bm25, a cross-encoder reranker, and the Anthropic SDK. No LlamaIndex, no LangChain.

---

## The Build Order

### Step 1-3: Corpus and Parsing
Parsed the LOTR epub and Wikipedia articles into clean JSON documents. The epub used `\r\n` word-wrapping with no real paragraph breaks — the parser had to extract `<p>` tags individually. The wiki corpus used single `\n` and required auto-detection of the delimiter. Getting clean text out of raw sources is harder than it sounds and everything downstream depends on it.

### Step 4: Embeddings and Vector Store
Embedded chunks with `all-MiniLM-L6-v2` (384 dimensions) and stored them in ChromaDB. Established the `embed()` abstraction that routes to OpenAI or sentence-transformers based on model name — so switching embedding models only requires a config change.

### Step 5: Hybrid Search (BM25 + Vector + RRF)
Added BM25 keyword search alongside vector search and merged results with Reciprocal Rank Fusion (RRF). Vector search finds semantically similar passages. BM25 finds exact names and quotes. Tolkien's world is full of proper nouns — Éowyn, Caradhras, Orthanc — that semantic search alone struggles with. Hybrid search was the single biggest retrieval improvement.

**RRF formula:** each result at rank `r` contributes `1 / (60 + r + 1)` to its score. Results appearing in both lists accumulate from both — they float to the top naturally.

### Step 6: Generation with Claude
Wired retrieval into Claude with a grounded system prompt: answer using ONLY the provided passages, say so clearly if the answer isn't there. The grounding rule is what makes RAG trustworthy — without it, Claude will fill gaps with plausible-sounding hallucinations.

### Step 7: Cross-Encoder Reranker
Added a cross-encoder reranker (a small model that sees query + passage together) to re-score the merged candidate pool before sending the top-k to Claude. A bi-encoder (used for embedding) scores query and passage independently and compares vectors. A cross-encoder sees both at once and is much more accurate — it just can't scale to the full corpus, which is why it runs on the top 20 candidates rather than all chunks.

### Step 8: Paragraph Chunking
Replaced fixed-size chunking with paragraph-aware chunking. Fixed-size chunking ignores sentence and paragraph boundaries — a chunk might start mid-sentence and end mid-thought. Tolkien writes one idea per paragraph more often than not, so paragraphs are the natural unit of semantic coherence. This was a large quality jump.

### Step 9: Wiki Corpus
Added a second corpus of Wikipedia articles covering LOTR lore. Some questions require general background knowledge (who is Sméagol, what are the Paths of the Dead) that isn't answered in any single passage of the novels. The wiki corpus fills those gaps. A `corpus=both` query searches both collections and merges the results before reranking.

### Step 10: Evaluation Framework
Built an LLM-as-judge eval system: 30 carefully designed questions with verified expected answers, scored 1-5 by Claude acting as judge. The judge returns a score, reasoning, and failure mode (retrieval vs generation). Key lessons:
- With 12 questions, ±0.4 point run-to-run variance makes improvements invisible — you need 30+
- Bad question design (false premises, overly narrow expected answers) creates phantom failures
- Retrieval failures and generation failures need different fixes — confusing them wastes effort

---

## Score Progression

All scores below are averages on a 1-5 scale judged by Claude. The first phase used a 12-question eval set; from query expansion onward we expanded to 30 questions, which is why the scores are not directly comparable across that boundary — 30 questions have lower variance and give a more reliable signal.

### Techniques that stuck (12-question eval)

| Step | Technique added | Score |
|---|---|---|
| Baseline | Fixed-size chunks, vector search only | 2.7 |
| +Hybrid search | BM25 + vector + RRF | 3.2 |
| +Reranker | Cross-encoder reranking of top-20 candidates | 3.5 |
| +Paragraph chunking | Semantically coherent chunks instead of fixed-size | 3.7 |
| +Wiki corpus | Wikipedia lore articles as a second retrieval source | 4.1 |

### Techniques tried and reverted (12-question eval)

All of these were evaluated against the 4.1 baseline and reverted.

| Technique | Score | Verdict |
|---|---|---|
| HyDE | 3.5 | Worse — hypothetical answers introduced retrieval bias |
| OpenAI text-embedding-3-small | 3.3 | Worse — not optimised for literary short-passage similarity |
| all-mpnet-base-v2 (768-dim) | 3.5 | Worse — larger local model ≠ better retrieval |
| Extraction prompt | 4.1 | Neutral — per-question variance cancelled out |
| Query expansion | 4.0 | Worse — more noise than signal at this corpus size |

### Eval expansion and tuning (30-question eval)

Expanded from 12 to 30 questions to get reliable signal. Several questions had false premises or imprecise phrasings that were producing phantom failures — fixing those was as valuable as any technique change.

| Step | Change | Score |
|---|---|---|
| First 30-question run | Baseline with new questions | 4.0 |
| Question phrasing fixes | Q023 (athelas), Q030 (Legolas snow) | 4.0 |
| More phrasing fixes | Q013 (Éowyn), Q018 (Bilbo butter) | 4.1 |
| +Parent-child chunking | Small children (30-120 words) + window=2 expansion | **4.2** |
| Window=3 experiment | Wider expansion window | 4.2 (more retrieval failures, reverted) |

---

## What We Tried and Reverted

Every technique below was implemented, evaluated, and reverted because it didn't improve the score.

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer to the query with Claude, embed that instead of the raw query. The idea: the hypothetical answer lives in the same embedding space as real passages, so it retrieves better than the question itself.

**Why it didn't help:** Q003 and Q009 regressed. Hypothetical answers introduce retrieval bias — if the model hallucinates a plausible-but-wrong answer, you retrieve passages about the wrong thing. Works better on factual corpora; less reliable on literary text.

### OpenAI text-embedding-3-small
Replaced `all-MiniLM-L6-v2` (384-dim) with OpenAI's `text-embedding-3-small` (1536-dim). Required batching during ingest to stay under OpenAI's 300k token-per-request limit.

**Why it didn't help:** Score dropped from 4.0 → 3.3. Higher-dimensional embeddings aren't automatically better — they're trained on different data with different objectives. `all-MiniLM-L6-v2` was specifically optimised for semantic similarity on short passages, which matches this task well.

### all-mpnet-base-v2
Tried a larger local model (768-dim) expecting it to outperform the smaller one.

**Why it didn't help:** Score dropped from 4.0 → 3.5. Model size ≠ retrieval quality. The smaller model was already well-suited to this task.

### Extraction Prompt
Added an instruction to the system prompt: "identify the most relevant sentence before answering." Intended to force Claude to locate the exact detail rather than paraphrase the passage.

**Why it didn't help:** Q006 dropped 4→3 while Q010 improved 3→4. Neutral on average, and the additional instruction adds noise for questions where the relevant detail is spread across multiple sentences rather than concentrated in one.

### Query Expansion
Generated 4 alternative phrasings of each query with Claude, retrieved for each phrasing independently, merged all results with RRF. The idea: a single query embeds near one region of the vector space; different phrasings surface different relevant chunks.

**Why it didn't help:** Q006 dropped 4→2, Q010 regressed. More retrievals = more noise in the candidate pool, which can push the right passage below the reranker's cutoff. Works better with larger corpora where coverage is the bottleneck; here, the corpus is small enough that the original query already finds the right passage when it exists.

**The lesson from all of these:** the fundamentals (chunking quality, hybrid search, reranking) did all the real work. Adding sophistication on top of a well-tuned baseline tends to move within noise or hurt, not help.

---

## Parent-Child Chunking

The final technique that did work.

### The Problem It Solves
After tuning, 5 of 30 questions were **generation failures** — the right chapter was retrieved but the key sentence was in an adjacent chunk that the boundary cut off. Claude had the right area but not the right paragraph.

Example: Q007 asked what Boromir says to Aragorn as he dies. The confession ("I tried to take the Ring") and the farewell ("Farewell, Aragorn! Go to Minas Tirith...") are in adjacent paragraphs. The old chunking retrieved the confession paragraph but not the farewell.

### How It Works

**Index phase:** split documents into small child chunks (30-120 words). Each child stores its paragraph index in the source document. These small, focused chunks are what get embedded.

**Query phase:** retrieve the best-matching child chunks as before. Then, before sending to Claude, expand each child by loading the source document and extracting ±2 surrounding paragraphs. Claude sees the expanded window; the embedding never did.

```
Child retrieved:  "'Farewell, Aragorn! Go to Minas Tirith and save my people!'"
                   [paragraph 47 of lotr_023.json]

Window sent to Claude:  paragraphs 45, 46, 47, 48, 49
                         = the full death scene including confession + farewell + Aragorn's response
```

### The Parameters

- **`MIN_WORDS=30`** — minimum words per child chunk. Smaller = more precise retrieval embeddings.
- **`MAX_WORDS=120`** — maximum words per child chunk. Keeps children focused.
- **`_WINDOW=2`** — paragraphs before and after to include in the context sent to Claude.

`_WINDOW` is applied at query time — no re-ingestion needed to change it. Window=3 was tested and produced more retrieval failures than window=2, confirming 2 as the right balance.

### What It Changed

| Metric | Before | After |
|---|---|---|
| Average score | 4.1 / 5.0 | 4.2 / 5.0 |
| Clean answers | 22 / 30 | 24 / 30 |
| Retrieval failures | 3 | 5 |
| Generation failures | 5 | 1 |

The score improvement is modest. The failure mode shift is the real result. Generation failures collapsed; retrieval failures increased slightly as a trade-off from smaller children. This is meaningful because retrieval failures are honest ("the passage wasn't found") while generation failures are confusing ("the passage was there but Claude still got it wrong"). A system that fails at retrieval is easier to reason about and fix.

---

## Final Results

**Score: 4.2 / 5.0 across 30 verified questions**
**24/30 questions answered correctly**

Remaining failures (all understood):
- **Retrieval (5):** Aragorn's names (multi-hop across trilogy), Éowyn's reveal (specific scene not retrieved), Fellowship roster (not in one chunk), Scouring of the Shire (appendix-style chunks), Glorfindel vs Arwen (false premise question)
- **Generation (1):** Frodo claiming the Ring at the Crack of Doom

---

## Techniques That Stuck

| Technique | Why it helped |
|---|---|
| Paragraph chunking | Semantically coherent units; Tolkien writes one idea per paragraph |
| Hybrid search (BM25 + vector + RRF) | Keywords catch proper nouns; vectors catch meaning; RRF merges cleanly |
| Cross-encoder reranker | Sees query + passage together; much more accurate than vector similarity alone |
| Wiki corpus | Fills lore gaps that the novels don't explain in a single passage |
| Parent-child (window=2) | Small children for precise retrieval; expanded window for complete generation |

---

## Things to Keep in Mind

**Chunking strategy is the highest-leverage decision.** Everything downstream depends on it. Get this right before tuning anything else.

**Eval quality matters more than eval size.** False-premise questions and wrong expected answers create phantom failures that waste diagnostic effort. Verify answers against the corpus before committing them.

**LLM-as-judge has variance.** With 12 questions, ±0.4 points run-to-run is noise. 30+ questions are needed for the signal to be trustworthy.

**Retrieval and generation failures need different fixes.** Retrieval failures: improve search or chunking. Generation failures: improve context (window size, prompt). Diagnosing which you have before trying a fix saves a lot of time.

**In production, `top_k` and `corpus` are fixed defaults.** The per-question tuning in the eval JSON is engineering knowledge, not product logic. A Next.js frontend just sends a question string; the defaults are applied internally.

---

## Is Narrative the Right Use Case for RAG?

Honestly, no — but it was a great one for learning.

### Why narrative is a hard fit

RAG works best when answers are localized: a fact, a definition, a procedure lives in one place, and retrieving that place gives you the answer. Technical documentation, legal contracts, support knowledge bases, medical literature — these are the natural home of RAG.

Narrative is the opposite. Meaning accumulates across the whole story. Frodo's relationship with the Ring isn't in any single passage — it's built across hundreds of pages of small moments. Character motivations, themes, emotional arcs — these require synthesis, not retrieval. This showed up directly in the eval: the hardest failures were almost always questions requiring connections across multiple passages (Aragorn's many names scattered across the trilogy, the Scouring of the Shire details spread across an entire chapter), while the easiest wins were isolated quotes and facts ("What does Frodo say at the Council of Elrond?").

### What RAG is genuinely good at on narrative

Specific quotes, scene details, and factual lookups do work well. "What does Bilbo say about feeling like butter?" is a retrieval task — there's one passage and it contains the answer. So RAG on narrative isn't useless, it's just limited to a subset of the questions you'd actually want to ask about a book.

### What would work better for narrative questions

Long-context models. A significant portion of LOTR would fit inside a 1M token context window — you could ask questions directly without retrieval at all. For a fixed, known corpus like a single book, that's often the right answer: skip RAG entirely. RAG becomes necessary when the corpus is too large for context, changes frequently, or spans many documents.

### So why was this a good project anyway

Because the difficulty of the use case forced you to confront real problems. Vocabulary mismatch, chunk boundary failures, eval design, retrieval vs generation failure modes — these all surfaced clearly precisely because narrative is unforgiving. A documentation RAG would have scored 4.5 out of the box and you'd have learned much less. The hard use case was a better teacher.

---

## What This Build Teaches You About Frameworks

LlamaIndex and LangChain give you all these components pre-built. But they also hide what each one does, make debugging hard, and add abstraction you have to fight when something goes wrong.

By building each piece explicitly you now know:
- Why hybrid search beats vector-only
- What RRF actually does and why it works
- What a cross-encoder does that a bi-encoder can't
- Why chunking strategy matters more than the embedding model
- How LLM-as-judge works and where it breaks down
- Why more complexity often doesn't mean better results

When you pick up LlamaIndex or LangChain next, you'll know what's inside the black box — which means you'll know when to use the default and when to override it.
