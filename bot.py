
import os
import time
import math
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv

# LangChain / Chroma imports (names you were using)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Message types from langchain_core (used to pass messages to many LangChain wrappers)
from langchain_core.messages import HumanMessage, SystemMessage

# --- Load Environment Variables ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# --- RAG Configuration ---
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-4o-mini"   # keep temp = 0 for accuracy
COLLECTION_NAME = "accurate_articles"

# Retrieval / tuning defaults (tune these per your dataset)
RETRIEVE_K = 20           # initial dense retrieval (candidates)
FINAL_PASS_K = 5          # how many final chunks to include in the LLM context
DEDUPE_SNIPPET_CHARS = 200
CONFIDENCE_SIM_THRESHOLD = 0.15   # heuristics; if low, answer "I don't know"
CACHE_TTL = 300          # seconds to cache identical queries

# --- Globals ---
CHROMA_DB = None
LLM = None
EMBEDDINGS = None

# Simple in-memory cache: {query_lower: (answer, expires_at, provenance)}
SIMPLE_CACHE = {}

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- High-Accuracy System Prompt (unchanged) ---
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

# -------------------------
# Initialization / Startup
# -------------------------
@bot.event
async def on_ready():
    global CHROMA_DB, LLM, EMBEDDINGS
    print(f"{bot.user} has connected to Discord — initializing RAG components.")

    # validate environment
    if not all([DISCORD_TOKEN, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
        print("FATAL: Missing one or more environment variables for Discord/Chroma.")
        await bot.close()
        return

    try:
        # instantiate embeddings
        EMBEDDINGS = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

        # create chroma client in executor (cloud)
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

        # init LLM (temperature 0 for deterministic answers)
        LLM = ChatOpenAI(model=OPENAI_GENERATION_MODEL, temperature=0)
        print("✅ ChatOpenAI client initialized.")

    except Exception as e:
        print("ERROR initializing RAG components:", e)
        await bot.close()

# -------------------------
# Utility functions
# -------------------------
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
    key = query.strip().lower()
    SIMPLE_CACHE[key] = (answer, time.time() + ttl, provenance)

def safe_extract_text_from_llm_resp(resp):
    """
    Defensive extractor for various LangChain return shapes:
    - direct string
    - object with .content
    - object with .generations -> nested
    - object with .text
    """
    if resp is None:
        return ""
    # string
    if isinstance(resp, str):
        return resp.strip()
    # common .content
    if hasattr(resp, "content"):
        try:
            return getattr(resp, "content").strip()
        except Exception:
            pass
    # generations (list or list of lists)
    if hasattr(resp, "generations"):
        gens = getattr(resp, "generations")
        try:
            # nested list
            first = gens[0]
            if isinstance(first, list):
                cand = first[0]
            else:
                cand = first
            # try known attributes
            if hasattr(cand, "text"):
                return cand.text.strip()
            if hasattr(cand, "message"):
                # some wrappers store message
                msg = getattr(cand, "message")
                if hasattr(msg, "content"):
                    return msg.content.strip()
                return str(msg).strip()
        except Exception:
            pass
    # fallback to str()
    try:
        return str(resp).strip()
    except Exception:
        return ""

def call_llm_sync(messages):
    """
    Call the ChatOpenAI wrapper in a *safe* way for multiple LangChain shapes.
    Tries several method names (.invoke, .predict_messages, .generate, __call__).
    Returns the extracted text.
    """
    global LLM
    if LLM is None:
        raise RuntimeError("LLM not initialized.")

    # Try a number of supported methods in order
    candidates = [
        ("invoke", lambda obj, arg: obj.invoke(arg)),
        ("predict_messages", lambda obj, arg: obj.predict_messages(arg)),
        ("generate", lambda obj, arg: obj.generate(arg)),
        ("__call__", lambda obj, arg: obj.__call__(arg)),
    ]

    last_exc = None
    for name, fn in candidates:
        try:
            resp = fn(LLM, messages)
            text = safe_extract_text_from_llm_resp(resp)
            if text:
                return text
            # if no text, continue to next candidate
        except TypeError as te:
            last_exc = te
            # callable may not exist - continue
        except AttributeError as ae:
            last_exc = ae
        except Exception as e:
            # some wrappers raise on unexpected shapes; capture and try next
            last_exc = e

    # If we reach here, nothing worked
    raise RuntimeError(f"Failed to call LLM (last error: {last_exc})")

# retrieve documents in executor; the function is defensive of retriever method names
def retrieve_docs_sync(query, k=RETRIEVE_K, collection_filter=None):
    """
    Synchronous retrieval helper (meant to run in executor).
    Returns a list of docs (objects) in the same shape LangChain usually uses: doc.page_content and doc.metadata.
    """
    global CHROMA_DB
    if CHROMA_DB is None:
        raise RuntimeError("Chroma DB not initialized.")

    # Build retriever
    # CHROMA_DB may be either a LangChain "VectorStore" or a custom wrapper with as_retriever()
    retriever_obj = None
    try:
        # If CHROMA_DB exposes as_retriever (LangChain pattern)
        if hasattr(CHROMA_DB, "as_retriever"):
            retriever_obj = CHROMA_DB.as_retriever(search_kwargs={"k": k})
        elif hasattr(CHROMA_DB, "get_relevant_documents"):
            # If the Chroma wrapper is itself a VectorStore-like object
            # we'll call get_relevant_documents directly below
            retriever_obj = CHROMA_DB
        else:
            retriever_obj = CHROMA_DB
    except Exception:
        retriever_obj = CHROMA_DB

    # Try different retrieval method names in order
    try_methods = [
        "get_relevant_documents",  # common
        "get_relevant_documents_for_query",  # some wrappers
        "retrieve",
        "get_documents",
        "as_retriever",  # unlikely but harmless
        "invoke",
        "__call__",
    ]

    # If retriever_obj was produced from as_retriever, it might already provide get_relevant_documents
    for m in try_methods:
        if hasattr(retriever_obj, m):
            meth = getattr(retriever_obj, m)
            try:
                # Many retrieve methods accept the query + optional k or search_kwargs
                # Try common signatures in order
                try:
                    return meth(query)  # simple call
                except TypeError:
                    try:
                        return meth(query, k=k)
                    except TypeError:
                        try:
                            return meth(query, search_kwargs={"k": k})
                        except TypeError:
                            # try with kwargs only
                            return meth(**{"query": query, "k": k})
            except Exception:
                # If that fails, try next method
                continue

    # final fallback: if CHROMA_DB provides a raw 'query' method
    if hasattr(CHROMA_DB, "query"):
        try:
            return CHROMA_DB.query(query, n_results=k)
        except Exception:
            pass

    raise RuntimeError("No suitable retriever method found on Chroma DB / retriever object.")

def dedupe_and_select(docs, final_k=FINAL_PASS_K):
    """
    Very simple dedupe based on source label + snippet to drop exact duplicates.
    Returns top `final_k` docs.
    """
    out = []
    seen = set()
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or meta.get("id") or "unknown"
        sig = source + "|" + (text[:DEDUPE_SNIPPET_CHARS])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(d)
        if len(out) >= final_k:
            break
    return out

# -------------------------
# Core LLM + RAG glue
# -------------------------
def build_context_from_docs(selected_docs):
    """Construct a single context string for the LLM that includes provenance per chunk."""
    pieces = []
    for idx, d in enumerate(selected_docs, start=1):
        text = getattr(d, "page_content", "") or ""
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{idx}"
        title = meta.get("heading") or meta.get("title") or ""
        piece = f"[{idx}] source={source} title={title}\n{text}"
        pieces.append(piece)
    return "\n\n---\n\n".join(pieces)

def generate_answer_sync(question, context_text):
    """Synchronous call that builds messages and calls LLM safely."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context_text)),
        HumanMessage(content=question)
    ]
    answer = call_llm_sync(messages)
    return answer

# -------------------------
# Discord command
# -------------------------
@bot.command(name="ask", help="Ask the bot a question based on the uploaded articles.")
async def ask_rag(ctx, *, question: str):
    global CHROMA_DB, LLM, EMBEDDINGS

    # quick pre-checks
    if CHROMA_DB is None or LLM is None or EMBEDDINGS is None:
        await ctx.send("Knowledge base not ready yet. Try again in a moment.")
        return

    # cache hit?
    cached = cache_get(question)
    if cached:
        answer, expires_at, provenance = cached
        # present cached answer with short provenance
        embed = discord.Embed(title="✅ Cached Answer", description=answer, color=0x22DD88)
        if provenance:
            embed.add_field(name="Provenance (cached)", value=provenance, inline=False)
        await ctx.send(embed=embed)
        return

    # start typing indicator
    async with ctx.typing():
        try:
            # 1) Retrieve candidate docs in executor
            docs = await bot.loop.run_in_executor(None, lambda: retrieve_docs_sync(question, k=RETRIEVE_K))

            if not docs:
                await ctx.send("No context found in the knowledge base for that question.")
                return

            # 2) Basic confidence / sanity check:
            # attempt to inspect doc scores if available (defensive)
            best_score = None
            try:
                # some doc shapes include .metadata['score'] or .score attribute
                first = docs[0]
                meta = getattr(first, "metadata", {}) or {}
                if "score" in meta:
                    best_score = float(meta.get("score"))
                elif hasattr(first, "score"):
                    best_score = float(getattr(first, "score"))
            except Exception:
                best_score = None

            # If best_score is present and very low, bail out with "I don't know"
            if best_score is not None and best_score < CONFIDENCE_SIM_THRESHOLD:
                # strict behavior: only return known phrase
                await ctx.send("**❓ Sorry, I cannot provide a definitive answer.**\n\n*The available articles do not contain the specific, figure-accurate information required to answer your question.*")
                return

            # 3) Dedupe + select top N
            selected = dedupe_and_select(docs, final_k=FINAL_PASS_K)

            # If after selection no docs, bail out
            if not selected:
                await ctx.send("No suitable content found to answer that question.")
                return

            # 4) Build context and generate answer (LLM call in executor)
            context_text = await bot.loop.run_in_executor(None, lambda: build_context_from_docs(selected))
            final_answer = await bot.loop.run_in_executor(None, lambda: generate_answer_sync(question, context_text))

            # Defensive normalization
            final_answer_text = (final_answer or "").strip()

            # If LLM returned exactly the required failure phrase, map to friendly message
            if final_answer_text.lower() == "i don't know the answer.":
                await ctx.send("**❓ Sorry, I cannot provide a definitive answer.**\n\n*The available articles do not contain the specific, figure-accurate information required to answer your question.*")
                return

            # Save to cache (store short provenance)
            provenance_summary = ", ".join([ (getattr(d, "metadata", {}) or {}).get("source") or (getattr(d, "metadata", {}) or {}).get("filename") or (getattr(d, "metadata", {}) or {}).get("doc_id") or f"doc_{i+1}" for i, d in enumerate(selected) ])
            cache_set(question, final_answer_text, provenance=provenance_summary)

            # 5) Respond with embed including short provenance excerpts
            embed = discord.Embed(title="✅ Accurate Answer", description=final_answer_text, color=0x22DD88)
            for idx, d in enumerate(selected, start=1):
                meta = getattr(d, "metadata", {}) or {}
                source_label = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{idx}"
                excerpt = (getattr(d, "page_content", "") or "")[:300]
                embed.add_field(name=f"Source [{idx}] — {source_label}", value=(excerpt + ("..." if len(getattr(d, "page_content", "") or "") > 300 else "")), inline=False)

            await ctx.send(embed=embed)
            return

        except Exception as exc:
            # Log server-side; return a friendly message
            print("RAG Processing Error:", exc)
            await ctx.send("An internal error occurred during RAG processing. Please check console logs.")
            return

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("DISCORD_BOT_TOKEN not set in environment.")
    else:
        bot.run(DISCORD_TOKEN)
