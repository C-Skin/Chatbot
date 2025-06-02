import chainlit as cl
from sentence_transformers import SentenceTransformer
import torch
import qdrant_client
from langchain.llms import Ollama
import logging
import time
from langdetect import detect

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Model & Client Initialization
# ----------------------------
model = SentenceTransformer('BAAI/bge-m3', device='cuda' if torch.cuda.is_available() else 'cpu')
client = qdrant_client.QdrantClient("http://35.188.22.78:6333")
llama = Ollama(model="llama3.1:8b", temperature=0.0, mirostat_tau=4.0, mirostat_eta=0.65)

# ----------------------------
# Function: Language Detection & Translation to English
# ----------------------------
def detect_and_translate(text):
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text
        else:
            translation_prompt = f"""
Translate the following sentence from {detected_lang} to English. Do not add explanation, just translate.

Original ({detected_lang}): {text}
English:
"""
            translated = llama.invoke(translation_prompt)
            return translated.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

# ----------------------------
# Function: Qdrant Search
# ----------------------------
def search(query):
    logger.info("Searching for relevant documents with query: %s", query)
    start_time = time.time()
    try:
        query_vector = model.encode(query).tolist()
        results = client.search(
            collection_name='Skin Diseases',
            query_vector=query_vector,
            limit=5,
            with_payload=True,
            score_threshold=0.4
        )
        elapsed = time.time() - start_time
        logger.info("Search completed in %.2f seconds", elapsed)

        # Format results
        sorted_result = sorted(results, key=lambda x: x.score, reverse=True)
        final_results = []
        for res in sorted_result:
            payload = res.payload
            combined = f"Q: {payload.get('question', '')}\nA: {payload.get('answer', '')}\nSource: {payload.get('source', '')}\nFocus Area: {payload.get('focus_area', '')}"
            final_results.append(combined.strip())
        return final_results
    except Exception as e:
        logger.error("Error while searching: %s", str(e))
        return []

# ----------------------------
# Function: Generate Response Using Ollama
# ----------------------------
def generate_response_with_ollama(context, query, tone="professional and friendly"):
    try:
        logger.info("Generating response using Ollama.")
        start_time = time.time()

        # Gabungkan semua konteks
        context_text = "\n\n".join([f"Doc {i+1}:\n{ctx.strip()}" for i, ctx in enumerate(context)])

        # Prompt Template
        prompt = f"""
Anda adalah chatbot kesehatan bernama C-Skin. Silakan jawab setiap pertanyaan pengguna dengan nada: {tone}.

Petunjuk untuk menjawab pertanyaan terkait penyakit:
- Jawablah hanya menggunakan informasi yang disediakan dalam {context}.
- Jika pertanyaan berkaitan dengan suatu penyakit, berikan informasi yang akurat, ringkas, dan lengkap berdasarkan konteks.
- Jika tidak ditemukan informasi yang relevan dalam konteks, jangan berspekulasi atau mengarang jawaban. Sebagai gantinya, balas dengan: "Ini adalah semua informasi yang saya miliki."
- Hindari penggunaan istilah medis yang kompleks atau tidak dikenal. Gunakan bahasa yang sederhana dan mudah dipahami oleh pengguna umum.
- Selalu mulai jawaban Anda dengan: "Terima kasih telah berkonsultasi dengan C-Skin."
- Selalu akhiri jawaban Anda dengan: "Semoga informasi ini bermanfaat, lekas sembuh, dan terima kasih."
- Jangan gunakan pengetahuan dari luar konteks yang diberikan.
- Jangan menyebutkan bahwa Anda terbatas oleh konteks — cukup sampaikan jawaban berdasarkan informasi yang tersedia.
- Anda tidak boleh menebak atau mengada-ada fakta. Hanya parafrase atau rangkum informasi yang secara eksplisit tersedia dalam konteks.

Gunakan hanya konteks berikut untuk menjawab pertanyaan pengguna:
{context_text}

Pertanyaan Pengguna:
{query}

Jawaban hanya dalam Bahasa Indonesia.
"""
        result = llama.invoke(prompt, stream=True)
        elapsed = time.time() - start_time
        logger.info("Inference completed in %.2f seconds", elapsed)
        return result
    except Exception as e:
        logger.error(f"Error while generating response with Ollama: {e}")
        return "Terjadi kesalahan saat menghasilkan jawaban. Silakan coba lagi."

# ----------------------------
# Async Wrappers
# ----------------------------
search_async = cl.make_async(search)
generate_response_async = cl.make_async(generate_response_with_ollama)

# ----------------------------
# Chainlit Message Handler
# ----------------------------
@cl.on_message
async def main(message: cl.Message):
    query = message.content
    query = detect_and_translate(query)
    logger.info("Received message content: %s", query)

    history = cl.user_session.get("history", [])
    history.append({"role": "user", "text": query})
    cl.user_session.set("history", history)

    # Sidebar removed

    results = await search_async(query)

    if not results:
        await cl.Message(content="Tidak ditemukan informasi yang relevan.").send()
        return

    response = await generate_response_async(results, query)

    history.append({"role": "assistant", "text": response})
    cl.user_session.set("history", history)

    await cl.Message(content=response).send()

# ----------------------------
# Handler: Resume Chat
# ----------------------------
@cl.on_chat_resume
async def on_chat_resume(thread):
    await start()  # Panggil ulang fungsi `start` untuk konsistensi tampilan
