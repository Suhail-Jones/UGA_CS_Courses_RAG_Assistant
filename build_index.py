from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from bs4 import BeautifulSoup


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
    # TODO: load sources.txt + the 5 extra pages, chunk, embed, build the FAISS index
    
    initialSources = [
        "https://bulletin.uga.edu/Program/Details/73962",
        "https://bulletin.uga.edu/UnivInfo/content/university-info-undergrad-graduation-requirements.html",
        "https://cs.uga.edu/bachelor-science-computer-science",
        "https://doubledawgs.uga.edu/ProgramDetails/11937",
        "https://doubledawgs.uga.edu/ProgramDetails/10080"
    ]

    #Function to remove uneccesary elements from sources' html
    def extract_text(html):
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['nav', 'header', 'footer', 'script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    docs: list[Document] = []
    with open('sources.txt', 'r') as file:
        for line in file:
            line = line.strip()
            response = requests.get(line, timeout=20, headers=headers)
            docs.append(Document(page_content=extract_text(response.text), metadata={"source": line}))

    for src in initialSources:
        response = requests.get(src, timeout=20, headers=headers)
        docs.append(Document(page_content=extract_text(response.text), metadata={"source": src}))
    print(f"Loaded {len(docs)} docs.")
    
    #Splits the doc list into smaller chunks of 1000 characters
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = textSplitter.split_documents(docs)
    print(f'Split sources into {len(all_splits)} chunks.')

    vectorstore = None
    # TODO: save it with vectorstore.save_local(INDEX_DIR)

# vectorstore is now ready to use either way
