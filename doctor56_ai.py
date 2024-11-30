
import faiss # type: ignore
import numpy as np # type: ignore
from pypdf import PdfReader # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

query = "The calcium is elevated but pth is normal what does this mean?"

# this is a very small and efficient languge model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("document_index.faiss")

# Load document names
document_names = []
with open("document_names.txt", "r") as f:
    document_names = [line.strip() for line in f]

# Step 1: User Query
print(f"User query: {query}")

# Step 2: Generate Query Embedding
query_embedding = embedding_model.encode([query], convert_to_tensor=False)
query_embedding = np.array(query_embedding).astype("float32")  # Ensure it's float32

# Step 3: Search the Index
k = 3  # Number of results to retrieve
distances, indices = index.search(query_embedding, k)
document_contents = []
for i, idx in enumerate(indices[0]):
    try: 
        document = document_names[idx]
        print(f"reference: {document}")
        if not document.lower().endswith(".pdf"):
            continue
        text = ""
        reader = PdfReader(document)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        document_contents.append(text)
    except Exception as e:
        print(e)
        print(f"Invalid document names: {document}")
context = "\n\n".join(document_contents)

# ask ai
prompt = f"""
You are an AI assistant. Use the information from the following documents to answer the question accurately.

Documents:
{context}

Question: {query}

Answer:
"""
import google.generativeai as genai # type: ignore
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
print(response.text)