
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# LangChain / Chroma imports (you used these names — keep them)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- Load Environment Variables ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# --- RAG Configuration ---
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "accurate_articles"

# Retrieval & tuning defaults (tune these)
RETRIEVE_K = 20       # initial dense retrieval
FINAL_PASS_K = 5      # after dedupe/MMR, how many chunks to pass to LLM
DEDUPE_BY_SOURCE = True

# --- Globals ---
CHROMA_DB = None
LLM = None

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

@bot.event
async def on_ready():
    """Initializes the RAG components when the bot starts."""
    global CHROMA_DB, LLM
    print(f'{bot.user} has connected to Discord!')
    print("--- Initializing RAG Components (Chroma Cloud) ---")

    # Check for required cloud credentials
    if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
        print("FATAL ERROR: Missing one or more CHROMA_CLOUD environment variables.")
        await bot.close()
        return

    try:
        # Prepare embeddings and LLM (these are lightweight objects)
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

        # Synchronous wrapper to create Chroma cloud client (run in executor)
        def load_chroma_cloud():
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                chroma_cloud_api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )

        CHROMA_DB = await bot.loop.run_in_executor(None, load_chroma_cloud)
        print("ChromaDB successfully loaded from the Cloud.")

        # Initialize the Chat model (zero temperature for accuracy)
        LLM = ChatOpenAI(model=OPENAI_GENERATION_MODEL, temperature=0)
        print("OpenAI LLM successfully initialized.")

    except Exception as e:
        print(f"ERROR: Failed to initialize RAG components: {e}")
        await bot.close()

# --- Utility: safe LLM call that handles different response shapes ---
def call_llm_sync(messages):
    """
    Call the ChatOpenAI model synchronously (so we can run in executor).
    The function is defensive about the return object shape across LangChain versions.
    """
    global LLM
    if LLM is None:
        raise RuntimeError("LLM not initialized.")

    # Many LangChain Chat models are callable with a list of messages:
    # resp = LLM(messages)
    resp = LLM(messages)

    # Now extract the text in a robust way:
    # - If resp is a simple string
    # - If resp has attribute 'content' (AIMessage)
    # - If resp has 'generations' structure (list of lists with .text)
    # - If resp has .generations[0].text
    if isinstance(resp, str):
        return resp.strip()

    if hasattr(resp, "content"):
        return getattr(resp, "content").strip()

    # Older/newer LangChain shapes:
    if hasattr(resp, "generations"):
        gens = getattr(resp, "generations")
        # gens might be a list of Generation objects or list-of-lists
        try:
            # try nested list
            first = gens[0]
            if isinstance(first, list):
                text = first[0].text
            else:
                text = first[0].text if hasattr(first[0], "text") else first.text
            return text.strip()
        except Exception:
            pass

    # Fallback: str()
    return str(resp).strip()


def get_rag_answer(question, context_text):
    """Synchronous wrapper used with run_in_executor to query the LLM."""
    # Build prompt messages
    sys = SystemMessage(content=SYSTEM_PROMPT.format(context=context_text))
    human = HumanMessage(content=question)
    messages = [sys, human]

    # Call the LLM synchronously (we call this from executor)
    answer = call_llm_sync(messages)
    return answer

# --- Helper: dedupe simple by filename+chunk text signature (tunable) ---
def dedupe_docs(docs):
    seen = set()
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        # form a small signature: source filename (if available) + small hash of content
        source = meta.get("source") or meta.get("filename") or meta.get("doc_id") or "unknown"
        snippet = (d.page_content or "")[:200]
        sig = f"{source}::{snippet}"
        if sig in seen:
            continue
        seen.add(sig)
        out.append(d)
    return out

# --- Discord command: /ask or !ask ---
@bot.command(name='ask', help='Ask the bot a question based on its articles.')
async def ask_rag(ctx, *, question):
    global CHROMA_DB, LLM
    if CHROMA_DB is None or LLM is None:
        await ctx.send("The knowledge base is not fully initialized. Please try again in a moment.")
        return

    # Let user know we're processing (simple message)
    status = await ctx.send(f"Thinking about: **{question}**... (Checking context for accuracy)")

    try:
        # 1. Retrieval (run in executor because many Chroma clients are sync)
        def retrieve():
            # Use the retriever API exposed by the Chroma object.
            # Preferred LangChain retriever method name: get_relevant_documents
            # If your version exposes `as_retriever` returning a retriever, use that instead.
            retriever = CHROMA_DB.as_retriever(search_kwargs={"k": RETRIEVE_K})
            # use get_relevant_documents if available
            if hasattr(retriever, "get_relevant_documents"):
                return retriever.get_relevant_documents(question)
            if hasattr(retriever, "retrieve"):
                return retriever.retrieve(question)
            # Fallback to call() / invoke() if your wrapper uses it (last resort)
            if hasattr(retriever, "get_relevant_documents"):
                return retriever.get_relevant_documents(question)
            if hasattr(retriever, "invoke"):
                return retriever.invoke(question)
            # If nothing works, raise an error to be caught below
            raise RuntimeError("Retriever does not expose a known retrieval method in this environment.")
        
        docs = await bot.loop.run_in_executor(None, retrieve)

        if not docs:
            await status.edit(content="No context found in the knowledge base for that question.")
            return

        # 2. Post-process retrieved docs: dedupe and keep top FINAL_PASS_K
        docs = dedupe_docs(docs)
        selected = docs[:FINAL_PASS_K]

        # Build a context string that includes file source metadata for provenance
        context_pieces = []
        for i, d in enumerate(selected, start=1):
            meta = getattr(d, "metadata", {}) or {}
            source_label = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{i}"
            heading = meta.get("heading") or meta.get("title") or ""
            excerpt = (d.page_content or "").strip()
            piece = f"[{i}] source={source_label} heading={heading}\n{excerpt}"
            context_pieces.append(piece)

        context_text = "\n\n---\n\n".join(context_pieces)

        # 3. Generation: call the LLM in executor
        final_answer = await bot.loop.run_in_executor(None, lambda: get_rag_answer(question, context_text))

    except Exception as e:
        print(f"RAG Processing Error: {e}")
        await status.edit(content="An internal error occurred during RAG processing. Please check the console logs.")
        return

    # 4. Final output with provenance in Discord: prefer embeds (cleaner)
    if final_answer.strip().lower() == "i don't know the answer.":
        await status.edit(content=f"**❓ Sorry, I cannot provide a definitive answer.**\n\n*The available articles do not contain the specific, figure-accurate information required to answer your question.*")
        return

    # Build an embed message with answer + short source list
    embed = discord.Embed(title="✅ Accurate Answer", description=final_answer, color=0x22DD88)
    # Add sources as fields (short)
    for idx, d in enumerate(selected, start=1):
        meta = getattr(d, "metadata", {}) or {}
        source_label = meta.get("source") or meta.get("filename") or meta.get("doc_id") or f"doc_{idx}"
        # Put a short excerpt of each source in the field (max ~1024 chars)
        excerpt = (d.page_content or "")[:300]
        embed.add_field(name=f"Source [{idx}] — {source_label}", value=excerpt + ("..." if len(d.page_content or "") > 300 else ""), inline=False)

    await status.edit(content=None, embed=embed)

# --- Run the Bot ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_BOT_TOKEN environment variable is not set.")
        print("Please set the DISCORD_BOT_TOKEN in your .env file.")
    else:
        bot.run(DISCORD_TOKEN)
