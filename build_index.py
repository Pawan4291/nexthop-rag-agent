from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup

urls = [
"https://nexthop.ai/about-us/",
"https://nexthop.ai/platforms/",
"https://nexthop.ai/software-portfolio/",
"https://nexthop.ai/optics-and-cables/",
"https://nexthop.ai/software-releases/",
"https://nexthop.ai/support-hub/",
"https://nexthop.ai/learning-hub/",
"https://nexthop.ai/join-us/",
"https://nexthop.ai/contact-us/",
"https://nexthop.ai/platforms/regulatory-and-compliance/"
]

documents = []

for url in urls:
    print("Loading:", url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    text = soup.get_text()
    documents.append(text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents(documents)

print("Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)

vectorstore.persist()

print("Vector DB created successfully")