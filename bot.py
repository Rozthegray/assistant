

import os
import time
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Your LangChain / Chroma imports (keep the package names you used)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------
# Environment & Config
# -----------------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-4o-mini"

COLLECTION_NAME = "accurate_articles"

# Tuning parameters
RETRIEVE_K_PER_QUERY = 6     # retrieve this many per paraphrase
PARAPHRASE_COUNT = 6         # produce N paraphrases for the question
FINAL_PASS_K = 5             # how many final chunks to pass to LLM
DEDUPE_SNIPPET_CHARS = 200
CACHE_TTL = 300              # seconds
CONFIDENCE_SIM_THRESHOLD = 0.12  # optional if retriever provides scores

# Globals
CHROMA_DB = None
LLM = None
EMBEDDINGS = None

# Simple cache for identical queries
SIMPLE_CACHE = {}

# High-accuracy system prompt (keeps strict behavior)
SYSTEM_PROMPT = """
You are an expert answer generation system. Your primary directive is to provide an answer
that is accurate and precise, potentially down to the last figure or specific detail.

1.  **Strict Rule:** You must answer the user's question **using ONLY the CONTEXT provided below.**
2.  **Accuracy Check:** If the context contains all the necessary figures and definitive information, provide the specific answer clearly.
3.  **Ambiguity Rule:** If the context is missing specific figures, or if the information is ambiguous, contradictory, or insufficient to provide a definite, figure-accurate answer, you MUST NOT speculate or use external knowledge.
4.  **Failure Response:** In case of insufficient context, your ONLY response must be: **"I don't know the answer."** Do not include any explanation or filler text with this phrase.

CONTEXT:
{context}
"""

# -----------------------
# Bot Setup
# -----------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------
# Startup / init
# -----------------------
@bot.event
async def on_ready():
    global CHROMA_DB, LLM, EMBEDDINGS
    print(f"{bot.user} connected — initializing RAG components...")

    # Check environment
    if not all([DISCORD_TOKEN, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
        print("FATAL: Missing environment variables for Chroma/Discord.")
        await bot.close()
        return

    try:
        EMBEDDINGS = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

        def load_chroma():
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=EMBEDDINGS,
                chroma_cloud_api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )

        CHROMA_DB = await bot.loop.run_in_executor(None, load_chroma)
        print("✅ Chroma Cloud DB loaded.")

        LLM = ChatOpenAI(model=OPENAI_GENERATION_MODEL, temperature=0)
        print("✅ ChatOpenAI initialized.")

    except Exception as e:
        print("ERROR initializing RAG components:", e)
        await bot.close()

# -----------------------
# Utilities
# -----------------------
def cache_get(query):
    key = query.strip().lower()
    item = SIMPLE_CACHE.get(key)
    if not item:
        return None
    answer, expires_at, provenance = item
    if time.time() > expires_at:
        SIMPLE_CACHE.pop(key, None)
        return None
    return item

def cache_set(query, answer, provenance=None, ttl=CACHE_TTL):
    SIMPLE_CACHE[query.strip().lower()] = (answer, time.time() + ttl, provenance)

def safe_text(resp):
    """Robust extractor for varied LLM return shapes."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if hasattr(resp, "content"):
        try:
            return getattr(resp, "content").strip()
        except Exception:
            pass
    if hasattr(resp, "generations"):
        gens = getattr(resp, "generations")
        try:
            first = gens[0]
            if isinstance(first, list):
                cand = first[0]
            else:
                cand = first
            if hasattr(cand, "text"):
                return cand.text.strip()
            if hasattr(cand, "message") and hasattr(cand.message, "content"):
                return cand.message.content.strip()
        except Exception:
            pass
    try:
        return str(resp).strip()
    except Exception:
        return ""

def call_llm_sync(messages):
    """
    Try multiple plausible ChatOpenAI method names so the code works across versions:
    - invoke(messages)
    - predict_messages(messages)
    - generate(messages)
    - __call__(messages)
    """
    global LLM
    if LLM is None:
        raise RuntimeError("LLM not initialized.")

    methods = [
        ("invoke", lambda obj, arg: obj.invoke(arg)),
        ("predict_messages", lambda obj, arg: obj.predict_messages(arg)),
        ("generate", lambda obj, arg: obj.generate(arg)),
        ("__call__", lambda obj, arg: obj.__call__(arg)),
    ]
    last_exc = None
    for name, fn in methods:
        try:
            resp = fn(LLM, messages)
            txt = safe_text(resp)
            if txt:
                return txt
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Failed to call LLM; last error: {last_exc}")

# -----------------------
# Retrieval helpers
# -----------------------
def retrieve_docs_sync_single(query, k=RETRIEVE_K_PER_QUERY):
    """
    Defensive retrieval call for one query string. Returns list of docs.
    Accepts multiple retriever shapes.
    """
    global CHROMA_DB
    if CHROMA_DB is None:
        raise RuntimeError("Chroma DB not initialized.")

    # prefer as_retriever
    try:
        if hasattr(CHROMA_DB, "as_retriever"):
            retriever = CHROMA_DB.as_retriever(search_kwargs={"k": k})
            # try common methods on the retriever
            if hasattr(retriever, "get_relevant_documents"):
                return retriever.get_relevant_documents(query)
            if hasattr(retriever, "retrieve"):
                return retriever.retrieve(query)
            if hasattr(retriever, "invoke"):
                return retriever.invoke(query)
            if hasattr(retriever, "__call__"):
                return retriever(query)
        # fallback: direct vectorstore methods
        if hasattr(CHROMA_DB, "get_relevant_documents"):
            return CHROMA_DB.get_relevant_documents(query)
        if hasattr(CHROMA_DB, "query"):
            # some wrappers provide query(query, n_results=..)
            try:
                return CHROMA_DB.query(query, n_results=k)
            except TypeError:
                return CHROMA_DB.query(query)
    except Exception as e:
        # bubble up for outer handler to log
        raise RuntimeError(f"Retrieval error: {e}")
    raise RuntimeError("No suitable retrieval method found on the Chroma object or retriever.")

def aggregate_and_dedupe(docs_iterable, final_k=FINAL_PASS_K):
    """
    docs_iterable: list of lists of docs (from multiple paraphrases)
    Deduplicate by source + snippet and return up to final_k docs preserving order of first appearance.
    """
    seen = set()
    out = []
    for docs in docs_iterable:
        for d in docs:
            text = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}
            # Use minimal provenance keys but do not rely on title: source/filename/doc_id if present
            source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or meta.get("id") or "unknown"
            sig = source + "|" + text[:DEDUPE_SNIPPET_CHARS]
            if sig in seen:
                continue
            seen.add(sig)
            out.append(d)
            if len(out) >= final_k:
                return out
    return out

# -----------------------
# Semantic Query Expansion
# -----------------------
def expand_query_sync(question, n=PARAPHRASE_COUNT):
    """
    Use the LLM to produce n short paraphrases / alternative phrasings of the input question.
    Returns a list of paraphrases (including the original question first).
    """
    system = SystemMessage(content="You are a helpful assistant that rewrites user questions into short alternate phrasings suitable for semantic retrieval. Produce concise paraphrases without adding or removing meaning.")
    human = HumanMessage(content=f"Produce {n} concise paraphrases (1-8 words each is fine) for the user's question. Return them as a numbered list only. Question: {question}")
    messages = [system, human]
    resp_text = call_llm_sync(messages)
    # Parse numbered lines — fallback to splitting by newline
    paras = []
    for line in resp_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading numbering like "1. " or "1)"
        import re
        m = re.match(r"^\s*\d+[\.\)]\s*(.*)$", line)
        if m:
            paras.append(m.group(1).strip())
        else:
            paras.append(line)
        if len(paras) >= n:
            break
    # Always include original question at front (best recall)
    if question.strip() not in paras:
        paras.insert(0, question.strip())
    # ensure unique and limit to n+1
    unique = []
    for p in paras:
        if p not in unique:
            unique.append(p)
        if len(unique) >= n + 1:
            break
    return unique

# -----------------------
# LLM-based Re-ranker
# -----------------------
def rerank_docs_sync(question, candidates):
    """
    Given a question and a list of candidate docs, ask the LLM to score/re-rank them.
    Returns the candidates in ranked order (most relevant first).
    This is a lightweight re-ranker:
      - send each candidate's short excerpt to LLM with an index,
      - ask LLM to return a comma-separated ranked list of indices (best -> worst).
    """
    # Build prompt
    pieces = []
    for idx, d in enumerate(candidates, start=1):
        text = (getattr(d, "page_content", "") or "").strip().replace("\n", " ")
        # truncate for prompt size
        text_short = text[:800]
        pieces.append(f"[{idx}] {text_short}")
    system = SystemMessage(content="You are a relevance assessor. Given a user question and numbered candidate excerpts, rank which excerpts are most relevant to answering the question.")
    human_text = "Question: " + question + "\n\nCandidates:\n" + "\n".join(pieces) + "\n\nInstruction: Return ONLY a comma-separated list of candidate indices in order from most relevant to least relevant. Example: 3,1,2"
    human = HumanMessage(content=human_text)
    messages = [system, human]
    resp = call_llm_sync(messages)
    # parse indices
    import re
    nums = re.findall(r"\d+", resp)
    if not nums:
        return candidates  # fallback: unchanged
    order = []
    used = set()
    for n in nums:
        i = int(n)
        if 1 <= i <= len(candidates) and i not in used:
            order.append(candidates[i - 1])
            used.add(i)
    # append any not mentioned (least relevant)
    for i, c in enumerate(candidates):
        if c not in order:
            order.append(c)
    return order

# -----------------------
# Build context & final generation
# -----------------------
def build_context(selected_docs):
    """
    Build context string from selected docs, including minimal provenance but not titles.
    We include only short provenance labels (source/doc_id) and the chunk text.
    """
    pieces = []
    for i, d in enumerate(selected_docs, start=1):
        text = getattr(d, "page_content", "") or ""
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{i}"
        # Avoid using titles/headings as primary signal — include them only if present but not used for retrieval
        piece = f"[{i}] source={source}\n{text}"
        pieces.append(piece)
    return "\n\n---\n\n".join(pieces)

def generate_answer_sync(question, context_text):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context_text)),
        HumanMessage(content=question)
    ]
    return call_llm_sync(messages)

# -----------------------
# Discord Command
# -----------------------
@bot.command(name="ask", help="Ask the bot a question based on uploaded articles.")
async def ask_rag(ctx, *, question: str):
    global CHROMA_DB, LLM, EMBEDDINGS

    if CHROMA_DB is None or LLM is None or EMBEDDINGS is None:
        await ctx.send("Knowledge base not ready. Try again shortly.")
        return

    # Check cache
    cached = cache_get(question)
    if cached:
        answer, exp, prov = cached
        embed = discord.Embed(title="✅ Cached Answer", description=answer, color=0x22DD88)
        if prov:
            embed.add_field(name="Provenance", value=prov, inline=False)
        await ctx.send(embed=embed)
        return

    # Inform user
    status_msg = await ctx.send(f"Thinking about: **{question}**... (checking all article content for accuracy)")

    try:
        # 1) Expand the question into paraphrases (run in executor)
        paraphrases = await bot.loop.run_in_executor(None, lambda: expand_query_sync(question, n=PARAPHRASE_COUNT))

        # 2) For each paraphrase retrieve docs (in executor) and aggregate
        retrieve_tasks = []
        for p in paraphrases:
            # for each paraphrase we call retrieval sync function in executor
            retrieve_tasks.append(bot.loop.run_in_executor(None, lambda q=p: retrieve_docs_sync_single(q, k=RETRIEVE_K_PER_QUERY)))
        # collect results
        docs_lists = await asyncio.gather(*retrieve_tasks)
        # docs_lists is list of lists of docs (one list per paraphrase)

        # if none found at all
        any_found = any(docs_lists)
        if not any_found:
            await status_msg.edit(content="No context found in the knowledge base for that question.")
            return

        # 3) Aggregate + dedupe top candidates
        aggregated = aggregate_and_dedupe(docs_lists, final_k=FINAL_PASS_K * 3)  # get more to give reranker space
        if not aggregated:
            await status_msg.edit(content="No suitable context found to answer that question.")
            return

        # 4) Re-rank with LLM to pick best FINAL_PASS_K
        reranked = await bot.loop.run_in_executor(None, lambda: rerank_docs_sync(question, aggregated))
        selected = reranked[:FINAL_PASS_K]

        # 5) Build context and final LLM generation (in executor)
        context_text = await bot.loop.run_in_executor(None, lambda: build_context(selected))
        final_answer = await bot.loop.run_in_executor(None, lambda: generate_answer_sync(question, context_text))
        final_answer = (final_answer or "").strip()

        # 6) If final answer equals the strict failure phrase, map to friendly message
        if final_answer.lower() == "i don't know the answer.":
            await status_msg.edit(content="**❓ Sorry, I cannot provide a definitive answer.**\n\n*The available articles do not contain the specific, figure-accurate information required to answer your question.*")
            return

        # 7) Cache + send result with provenance (short)
        provenance = ", ".join([ (getattr(d, "metadata", {}) or {}).get("source") or (getattr(d, "metadata", {}) or {}).get("filename") or (getattr(d, "metadata", {}) or {}).get("doc_id") or f"doc_{i+1}" for i, d in enumerate(selected) ])
        cache_set(question, final_answer, provenance=provenance)

        embed = discord.Embed(title="✅ Accurate Answer", description=final_answer, color=0x22DD88)
        for idx, d in enumerate(selected, start=1):
            meta = getattr(d, "metadata", {}) or {}
            source_label = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{idx}"
            excerpt = (getattr(d, "page_content", "") or "")[:300]
            embed.add_field(name=f"Source [{idx}] — {source_label}", value=(excerpt + ("..." if len(getattr(d, "page_content", "") or "") > 300 else "")), inline=False)

        await status_msg.edit(content=None, embed=embed)
    except Exception as e:
        print("RAG Processing Error:", e)
        await status_msg.edit(content="An internal error occurred during RAG processing. Check the console logs.")
        return

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("DISCORD_BOT_TOKEN not set.")
    else:
        bot.run(DISCORD_TOKEN)
