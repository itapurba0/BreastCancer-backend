import os
import time
import warnings
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"
from qdrant_client import QdrantClient
from fastembed import TextEmbedding  
from openai import AsyncOpenAI

from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COLLECTION_NAME = "medical_knowledge_base"

# Mute warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

print("Loading lightweight embedding model...")
# Initialize FastEmbed (Uses ONNX runtime instead of PyTorch)
embed_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    cache_dir="/tmp/fastembed_cache"
)

# Connect to Qdrant
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
    print("✅ Connected to Qdrant")
except Exception as e:
    print(f"❌ Qdrant Connection Error: {e}")
    qdrant = None

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def retrieve_context(user_question: str, top_k: int = 1):
    if not qdrant:
        return ""

    try:
        start = time.time()

        # FastEmbed generates a generator, so we cast to list and grab the first item
        query_vector = list(embed_model.embed([user_question]))[0].tolist()

        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=0.5,  # Optional: filter out low-relevance results
            with_payload=True
        )


        context_chunks = []
        for hit in search_response.points:
            if hit.payload and "text" in hit.payload:
                context_chunks.append(hit.payload["text"][:1200])
        
        return "\n\n---\n\n".join(context_chunks)
    except Exception as e:
        print(f"❌ Qdrant Retrieval Error: {e}")
        return ""

# Change the function to act as a generator
# 3. NEW: Add 'async' to the function definition
async def generate_rag_response(messages):
    # --- BULLETPROOF DATA EXTRACTION ---
    if isinstance(messages, str):
        latest_user_question = messages
        ai_memory = [{"role": "user", "content": messages}]
    else:
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            latest_user_question = last_msg.get("content", "")
        else:
            latest_user_question = getattr(last_msg, "content", str(last_msg))

        ai_memory = []
        for msg in messages:
            if isinstance(msg, dict):
                ai_memory.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            else:
                ai_memory.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})
    # ------------------------------------

   
    context = retrieve_context(latest_user_question)
    if not context.strip():
        yield "I'm sorry, but I don't have enough information in my current medical files to answer that safely."
        return

    system_prompt = f"""
    You are a warm, empathetic, and supportive Breast Cancer Patient Navigator. 
    Your goal is strictly to spread awareness and help people understand breast cancer based ONLY on the provided documents.
    
    STRICT SCOPE GUARDRAILS:
    1. SCOPE: You are a BREAST CANCER companion ONLY. 
    2. OUT-OF-SCOPE RULE: If the user asks about any other type of cancer (e.g., brain cancer, lung cancer, leukemia) or any non-breast-cancer topic, you MUST politely decline to answer. 
    3. State clearly that you are specialized exclusively in breast cancer awareness and support, and gently guide them back to asking about breast diagnosis, screening, or treatments.
    
    STRICT TONE & PERSONA RULES:
    1. DO NOT sound like a doctor, oncologist, or medical student. 
    2. Speak to the user as if you are a supportive guide walking them through an awareness brochure.
    3. Use extremely simple, beginner-friendly language (8th-grade reading level).
    4. ONLY use the provided medical context. Do not pull in outside medical knowledge.
    
    FORMATTING: 
    - Use Markdown formatting (bullet points, bold text).
    - Keep your paragraphs short so they are easy to read on a screen.
    
    MEDICAL CONTEXT:
    {context}
    """

    ai_memory.insert(0, {"role": "system", "content": system_prompt})


    try:
        # 4. NEW: Add 'await' so Python knows it can handle other users while OpenRouter thinks
        # Request the stream with strict consistency parameters
        stream = await ai_client.chat.completions.create(
            model="openrouter/auto-route-free", # 1. Force a specific model instead of openrouter/free
            messages=ai_memory,
            temperature=0.0, 
            stream=True
        )

        # 5. NEW: Change 'for' to 'async for' to handle the asynchronous stream
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                # print(content, end="", flush=True)
                yield content
            
        print("\n")

    except Exception as e:
        print(f"\n❌ OpenRouter Error: {e}")
        yield "System Error: Unable to connect to the cloud AI engine."
