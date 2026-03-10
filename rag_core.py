import os
import time
import torch
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

# ChromaDB setup

# Use project-local Chroma folder
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_med_nut_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Main collection: modern medicine + nutrition
COLLECTION_NAME = "medical_nutrition_chatbot"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# Ayurveda collection (separate, no mixing)
AYURVEDA_COLLECTION_NAME = "ayurveda_chatbot"
ayurveda_collection = chroma_client.get_or_create_collection(
    name=AYURVEDA_COLLECTION_NAME,
    embedding_function=embedding_function
)

# Seed a few docs in the main collection if empty (first run only)
if collection.count() == 0:
    seed_docs = [
        "Some antibiotics can cause allergic reactions, including skin rashes. Patients should contact a doctor if they notice hives, itching, or swelling after starting an antibiotic.",
        "A balanced diet with fruits, vegetables, whole grains, and lean protein supports gut health and immune function.",
        "Drinking enough water every day is important for kidney function, digestion, and overall health."
    ]
    seed_ids = ["doc1", "doc2", "doc3"]
    seed_meta = [
        {"source": "seed", "topic": "antibiotics"},
        {"source": "seed", "topic": "nutrition"},
        {"source": "seed", "topic": "hydration"},
    ]
    collection.add(documents=seed_docs, ids=seed_ids, metadatas=seed_meta)
    print("[INFO] Seeded medical_nutrition_chatbot with a few example docs.")


# Remote LLM via Hugging Face (Mistral)

HF_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if HF_API_TOKEN is None:
    raise RuntimeError("Please set HUGGINGFACEHUB_API_TOKEN environment variable.")

HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

hf_client = InferenceClient(
    model=HF_MODEL_NAME,
    token=HF_API_TOKEN,
)


def generate_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Call hosted Mistral (conversational) via Hugging Face Inference API.
    """
    completion = hf_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.9,
    )
    return completion.choices[0].message["content"].strip()


# Allow TF32 where available (harmless to keep)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Shared RAG helpers

SYSTEM_PROMPT = """
You are a cautious medical, nutrition, and Ayurveda assistant.

You ONLY answer based on the provided context.
If the answer is not in the context, say “I don’t know based on this information.”
Never make up medical facts.

Use very simple language.
Answer briefly: at most 4–5 sentences or 3–5 short bullet points.
""".strip()




def format_docs(docs, max_chars: int = 800) -> str:
    """
    Join retrieved chunks into one context string,
    truncating if it gets too long.
    """
    text = ""
    for i, d in enumerate(docs):
        block = f"[Doc {i+1}]\n{d}\n\n"
        if len(text) + len(block) > max_chars:
            break
        text += block
    return text.strip()


def build_prompt(question: str, docs_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{docs_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# Main RAG: modern medicine + nutrition

def answer_with_rag(question: str, k: int = 3, max_new_tokens: int = 256) -> str:
    """
    Retrieve from the medical/nutrition collection and answer using Mistral.
    """
    # 1. Retrieve
    start_retrieval = time.time()
    result = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents"],
    )
    end_retrieval = time.time()
    print(f"[DEBUG][MED/NUT] Retrieval time: {end_retrieval - start_retrieval:.2f} seconds")

    docs = result["documents"][0] if result["documents"] else []

    if not docs:
        return "I could not retrieve any relevant information. Please try rephrasing your question."

    # 2. Build context and prompt
    docs_text = format_docs(docs)
    prompt = build_prompt(question, docs_text)

    # 3. Generate answer
    start_llm = time.time()
    answer = generate_llm(prompt, max_new_tokens=max_new_tokens)
    end_llm = time.time()
    print(f"[DEBUG][MED/NUT] LLM generation time: {end_llm - start_llm:.2f} seconds")

    return answer


# Ayurveda RAG: separate collection

def answer_with_rag_ayurveda(question: str, k: int = 3, max_new_tokens: int = 256) -> str:
    """
    Retrieve from the Ayurveda collection and answer using Mistral.
    """
    # 1. Retrieve
    start_retrieval = time.time()
    result = ayurveda_collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents"],
    )
    end_retrieval = time.time()
    print(f"[DEBUG][AYUR] Retrieval time: {end_retrieval - start_retrieval:.2f} seconds")

    docs = result["documents"][0] if result["documents"] else []

    if not docs:
        return "I could not retrieve relevant Ayurvedic information. Please try rephrasing your question."

    # 2. Build context and prompt
    docs_text = format_docs(docs)
    prompt = build_prompt(question, docs_text)

    # 3. Generate answer
    start_llm = time.time()
    answer = generate_llm(prompt, max_new_tokens=max_new_tokens)
    end_llm = time.time()
    print(f"[DEBUG][AYUR] LLM generation time: {end_llm - start_llm:.2f} seconds")

    return answer
