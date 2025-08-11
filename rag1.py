import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PDFRAGPipeline:
    def __init__(self, pdf_path, embed_model="all-MiniLM-L6-v2", gen_model="google/flan-t5-base"):
        self.pdf_path = pdf_path
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.embedder = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(gen_model)
        self.rag_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)

        self.text = self.load_pdf_text()
        self.chunks = self.chunk_text(self.text)
        self.embeddings = self.embed_chunks(self.chunks)
        self.index = self.build_faiss_index(self.embeddings)

    @staticmethod
    def disable_gpu():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def load_pdf_text(self):
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text

    @staticmethod
    def chunk_text(text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def embed_chunks(self, chunks):
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        return self.normalize(embeddings)

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Using cosine similarity
        index.add(embeddings)
        return index

    def retrieve(self, query, k=3, threshold=0.3):
        query_embedding = self.normalize(self.embedder.encode([query], convert_to_numpy=True))
        distances, indices = self.index.search(query_embedding, k)

        print("\n🔍 Top retrieved chunks and their similarity scores:")
        selected_chunks = []
        for i, score in zip(indices[0], distances[0]):
            print(f"Score: {score:.4f} | Chunk: {self.chunks[i][:100]}...")
            if score > threshold:
                selected_chunks.append(self.chunks[i])
        return selected_chunks

    def generate_answer(self, context, query):
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        output = self.rag_pipeline(prompt, max_new_tokens=100)
        return output[0]['generated_text']


def main():
    PDFRAGPipeline.disable_gpu()
    pdf_path = "heart_health_guide.pdf"  # Change to your file path
    rag = PDFRAGPipeline(pdf_path)

    while True:
        query = input("Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        top_chunks = rag.retrieve(query)

        if not top_chunks:
            print("📄 No relevant context found in the document.")
            print("💡 Answering from general knowledge...\n")
            answer = rag.generate_answer("", query)
        else:
            context = "\n".join(top_chunks)
            answer = rag.generate_answer(context, query)

        print("\n✅ Answer:\n", answer)


if __name__ == "__main__":
    main()
