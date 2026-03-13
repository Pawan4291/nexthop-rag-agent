from dotenv import load_dotenv
import os

load_dotenv()
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# -----------------------------
# 1. SCRAPE WEBSITE PAGES
# -----------------------------

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

print("Loaded pages:", len(documents))


# -----------------------------
# 2. SPLIT TEXT INTO CHUNKS
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents(documents)

print("Chunks created:", len(chunks))


# -----------------------------
# 3. CREATE VECTOR DATABASE
# -----------------------------

vectorstore = Chroma.from_documents(
    chunks
)

retriever = vectorstore.as_retriever()

print("Vector database ready")


# -----------------------------
# 4. CONNECT GROQ MODEL
# -----------------------------

import os

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# 5. QUESTION ANSWERING
# -----------------------------

def ask(question):

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAI Answer:\n")
    print(response.content)


# -----------------------------
# 6. INTERACTIVE CHAT
# -----------------------------

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/ask")
def ask_api(question: str):

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)