from dotenv import load_dotenv
load_dotenv()

import os, faiss
print("faiss:", faiss.__version__)
print("GOOGLE_API_KEY set:", bool(os.getenv("GOOGLE_API_KEY")))
print("LANGSMITH_API_KEY set:", bool(os.getenv("LANGSMITH_API_KEY")))

from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("embedding dim:", len(emb.embed_query("test")))

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
print("gemini says:", llm.invoke("Say OK").content.strip())