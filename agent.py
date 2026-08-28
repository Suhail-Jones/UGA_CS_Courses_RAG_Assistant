from dotenv import load_dotenv
import os

import uuid
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddingModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = "faiss_index"
vectorStore = FAISS.load_local(INDEX_DIR, embeddingModel, allow_dangerous_deserialization=True)


backend = StateBackend

@tool(parse_docstring = True)
def search_documentation(query):
    """Search the UGA CS Major documentation and save the most relevent chunks to the agent file system.

    Args:
        query: Natural language search query.

    Returns:
        Filepaths where retrieved chunks were saved under /retrieved/.
    """

    retrievedDocs = vectorStore.similarity_search(query=query, k=8)
    batch_id = uuid.uuid4().hex[:8]
    uploads = []
    savedPaths = []

    for index, doc in enumerate(retrievedDocs, start = 1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )