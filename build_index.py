from dotenv import load_dotenv
import os

import requests
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


headers = {
    "User-Agent": "UGA-CS-RAG-Assistant/0.1 (student project; contact: https://github.com/Suhail-Jones)"
}

INDEX_DIR = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
    print(f"Found existing index at '{INDEX_DIR}/' — loading instead of rebuilding.")
    # TODO: load the FAISS index from disk using `embeddings`
    vectorstore = None
else:
    print(f"No index found at '{INDEX_DIR}/' — building from scratch.")
    # TODO: load sources.txt + the 3 extra pages, chunk, embed, build the FAISS index
    
    docs: list[Document] = []
    with open('sources.txt', 'r') as file:
        for line in file:
            line = line.strip()
            response = requests.get(line, timeout=20, headers=headers)
            docs.append(Document(page_content=response.text, metadata={"source": line}))
    print(f"Loaded {len(docs)} docs.")

    vectorstore = None
    # TODO: save it with vectorstore.save_local(INDEX_DIR)

# vectorstore is now ready to use either way
