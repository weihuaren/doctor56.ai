import os
from pypdf import PdfReader # type: ignore
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

rootdir = b'resources/documents'
document_names = []
document_contents = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        try:
            document_names.append(os.path.join(subdir, file).decode("utf-8"))
        except Exception as e:
            pass
            # print("Invalid filename:")
            # print(os.path.join(subdir, file))
            # print(os.path.join(subdir, file) + e)

for document in document_names:
    print(f"processing: {document}")
    try: 
        if not document.lower().endswith(".pdf"):
            continue
        text = ""
        reader = PdfReader(document)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        document_contents.append(text)
    except Exception as e:
        print(f"Invalid document names: {document}")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # type: ignore

# Generate embeddings for each document
embeddings = model.encode(document_contents, convert_to_tensor=False) # type: ignore

# Create a FAISS index and add embeddings
dimension = embeddings.shape[1]  # Embedding dimensionality
index = faiss.IndexFlatL2(dimension)  # type: ignore # L2 distance for similarity
index.add(np.array(embeddings))  # type: ignore # Add document embeddings to the index
print(f"Added {len(embeddings)} documents to the index.")

faiss.write_index(index, "document_index.faiss")
print("FAISS index saved as 'document_index.faiss'.")

# Save document names for later retrieval
with open("document_names.txt", "w") as f:
    for name in document_names:
        f.write(f"{name}\n")
print("Document names saved as 'document_names.txt'.")

