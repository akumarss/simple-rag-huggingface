from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from PyPDF2 import PdfReader

# Disable GPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Step 1: Load PDF and extract text
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Normalize vectors
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Step 4: Build FAISS cosine index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity
    index.add(embeddings)
    return index

# Step 5: Embed chunks
def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return normalize(embeddings)

# Step 6: Retrieve top chunks
def retrieve(query, embedder, index, chunks, k=3, threshold=0.3):
    query_embedding = normalize(embedder.encode([query], convert_to_numpy=True))
    distances, indices = index.search(query_embedding, k)

    print("\n🔍 Top retrieved chunks and their similarity scores:")
    selected_chunks = []
    for i, score in zip(indices[0], distances[0]):
        print(f"Score: {score:.4f} | Chunk: {chunks[i][:100]}...")
        if score > threshold:
            selected_chunks.append(chunks[i])
    return selected_chunks

# Step 7: Generate answer
def generate_answer(context, query, rag_pipeline):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = rag_pipeline(prompt, max_new_tokens=100)
    return output[0]['generated_text']

# 🔄 Main
if __name__ == "__main__":
    pdf_path = "heart_health_guide.pdf"  # update with your file
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, embedder)
    index = build_faiss_index(embeddings)

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    rag_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

    while True:
        query = input("Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        top_chunks = retrieve(query, embedder, index, chunks, threshold=0.3)

        if not top_chunks:
            print("📄 No relevant context found in the document.")
            print("💡 Answering from general knowledge...\n")
            answer = generate_answer("", query, rag_pipeline)
        else:
            context = "\n".join(top_chunks)
            answer = generate_answer(context, query, rag_pipeline)

        print("\n✅ Answer:\n", answer)
