import os
import discord
from discord.ext import commands
# We are using the correct, modern langchain_chroma package now
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# ... other imports
from langchain_core.messages import HumanMessage, SystemMessage
# ...

# --- Load Environment Variables ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# NEW: Chroma Cloud Credentials
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# --- RAG Configuration ---
# CHROMA_PATH is no longer needed for cloud connection
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-4o-mini" 
COLLECTION_NAME = "accurate_articles" # The collection must exist in your Chroma Cloud DB

# --- RAG Setup (Global Variables) ---
CHROMA_DB = None
LLM = None

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- High-Accuracy System Prompt (Unchanged) ---
SYSTEM_PROMPT = """
You are an expert answer generation system. Your primary directive is to provide an answer 
that is accurate and precise, potentially down to the last figure or specific detail.

1.  **Strict Rule:** You must answer the user's question **using ONLY the CONTEXT provided below.**
2.  **Accuracy Check:** If the context contains all the necessary figures and definitive information, provide the specific answer clearly.
3.  **Ambiguity Rule:** If the context is missing specific figures, or if the information is ambiguous, contradictory, or insufficient to provide a definite, figure-accurate answer, you MUST NOT speculate or use external knowledge.
4.  **Failure Response:** In case of insufficient context, your ONLY response must be: **"I don't know the answer."** Do not include any explanation or filler text with this phrase.

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
        # Load the Chroma database from the Cloud
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        
        # Define the cloud connection function (still synchronous)
        def load_chroma_cloud():
            return Chroma(
                collection_name=COLLECTION_NAME, 
                embedding_function=embeddings,
                # --- CLOUD CONNECTION PARAMETERS ---
                chroma_cloud_api_key=CHROMA_API_KEY, 
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE
                # -----------------------------------
            )
        
        CHROMA_DB = await bot.loop.run_in_executor(None, load_chroma_cloud)
        print("ChromaDB successfully loaded from the Cloud.")

        # Initialize the OpenAI Chat Model
        LLM = ChatOpenAI(model=OPENAI_GENERATION_MODEL, temperature=0)
        print("OpenAI LLM successfully initialized.")

    except Exception as e:
        print(f"ERROR: Failed to initialize RAG components: {e}")
        await bot.close()


# Function to encapsulate the synchronous LLM call logic (Unchanged)
# Function to encapsulate the synchronous LLM call logic
def get_rag_answer(question, context_text):
    """Synchronous function to perform the LLM call."""
    global LLM
    if LLM is None:
        raise RuntimeError("LLM not initialized.")
        
    # --- FIX: We must pass a list of messages, not the template object ---
    
    # 1. Compile the messages using the SYSTEM_PROMPT format
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context_text)),
        HumanMessage(content=question)
    ]
    
    # 2. Invoke the LLM with the list of messages
    # This replaces LLM.invoke(prompt) which caused the error
    response = LLM.invoke(messages)
    return response.content.strip()

# NOTE: The rest of the bot.py file (including the main `ask_rag` command) remains the same.

@bot.command(name='ask', help='Ask the bot a question based on its articles.')
async def ask_rag(ctx, *, question):
    """Handles the user question, retrieves context, and generates a strict answer."""
    
    global CHROMA_DB, LLM
    if CHROMA_DB is None or LLM is None:
        await ctx.send("The knowledge base is not fully initialized. Please try again in a moment.")
        return

    await ctx.send(f"Thinking about: **{question}**... (Checking context for accuracy)")

    try:
        # 1. Retrieval: Execute the retrieval (network I/O to Chroma Cloud)
        retriever = CHROMA_DB.as_retriever(search_kwargs={"k": 5})
        
        # Retrieval is still synchronous (blocking) and must run in an executor
        docs = await bot.loop.run_in_executor(None, lambda: retriever.invoke(question))

        context_text = "\n---\n".join([doc.page_content for doc in docs])

        # 2. Generation: Execute the LLM call (network I/O to OpenAI)
        final_answer = await bot.loop.run_in_executor(None, lambda: get_rag_answer(question, context_text))
        
    except Exception as e:
        print(f"RAG Processing Error: {e}")
        await ctx.send("An internal error occurred during RAG processing. Please check the console logs.")
        return
    
    # 3. Final Output to Discord (Unchanged)
    if final_answer.lower() == "i don't know the answer.":
        await ctx.send(f"**❓ Sorry, I cannot provide a definitive answer.**\n\n*The available articles do not contain the specific, figure-accurate information required to answer your question.*")
    else:
        await ctx.send(f"**✅ Accurate Answer:**\n\n{final_answer}")


# --- Run the Bot ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_BOT_TOKEN environment variable is not set.")
        print("Please set the DISCORD_BOT_TOKEN in your .env file.")
    else:
        bot.run(DISCORD_TOKEN)