#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import logging
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List, Optional
from PyPDF2 import PdfReader

import google.generativeai as genai
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from IPython.display import Markdown

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("blog_generator.log"), logging.StreamHandler(sys.stdout)],
)

# Load environment variables if needed
load_dotenv()

# Set your API key (replace with your actual API key)
API_KEY = "AIzaSyB6DGLiB3XWLXqGrTzGDoGvnD2QOu8cvgc"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- BLOG & WEB SEARCH FUNCTIONS ---
def scrape_url(url: str) -> str:
    """Scrape the URL and return text content (up to 5000 characters)."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return text[:5000]
        else:
            return f"Failed to retrieve content. Status Code: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def generate_response(question: str, context: str) -> str:
    """Generate an answer based on the provided question and context."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

def search_duckduckgo(query: str, num_results: int = 5) -> List[str]:
    """Search DuckDuckGo for the query and return formatted search results."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append(f"- {r['title']} (URL: {r['href']})")
        return results or ["No search results found."]
    except Exception as e:
        logging.error(f"Search error: {e}")
        return ["Search failed."]

def generate_blog(topic: str, search_results: List[str]) -> Optional[str]:
    """Generate a blog post on the given topic using search results for context."""
    try:
        search_text = "\n".join(search_results)
        prompt = f"""
Write a comprehensive, engaging blog post on the topic: {topic}

Context from search results:
{search_text}

Guidelines:
- Create a well-structured blog with clear sections
- Include an attention-grabbing introduction
- Provide in-depth, informative content
- Use subheadings to break up text
- Conclude with key takeaways or insights
- Aim for approximately 800-1200 words
- Write in a conversational yet professional tone
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Blog generation error: {e}")
        return None

def save_blog(content: str, topic: str) -> str:
    """Save the generated blog content to a text file."""
    os.makedirs("blogs", exist_ok=True)
    filename = "".join(c if c.isalnum() or c.isspace() else "" for c in topic).replace(" ", "").lower()
    filepath = os.path.join("blogs", f"{filename}_blog.txt")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Blog saved successfully: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving blog: {e}")
        return ""

# --- PDF QA FUNCTIONS ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using the Gemini model."""
        result = genai.embed_content(
            model="models/embedding-001",
            content=input,
            task_type="retrieval_document",
            title="Systeme de management de l'environnement"
        )
        return result["embedding"]

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file, skipping the first 7 pages."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages[7:]:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def clean_extracted_text(text: str) -> str:
    """Clean the extracted text by removing unwanted characters and early content."""
    cleaned_text = ""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if len(line) > 10 and i > 70:
            cleaned_text += line + '\n'
    for char in ['.', '~', '©', '_', ';:;']:
        cleaned_text = cleaned_text.replace(char, '')
    return cleaned_text

def create_chroma_db(documents: List[str], name: str):
    """Create a Chroma database and add documents."""
    # Update the path as needed for your system
    chroma_client = chromadb.PersistentClient(path=r"C:\Users\CHARUMATHI N\Documents\New folder (2)")
    db = chroma_client.get_or_create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    initial_size = db.count()
    for i, doc in tqdm(enumerate(documents), total=len(documents), desc="Creating Chroma DB"):
        db.add(documents=doc, ids=str(i + initial_size))
        time.sleep(0.5)
    return db

def get_chroma_db(name: str):
    """Retrieve an existing Chroma database collection."""
    chroma_client = chromadb.PersistentClient(path=r"C:\Users\CHARUMATHI N\Documents\New folder (2)")
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

def get_relevant_passages(query: str, db, n_results: int = 5) -> List[str]:
    """Query the Chroma database and return relevant passages."""
    results = db.query(query_texts=[query], n_results=n_results)
    # Return the first set of documents if available; otherwise, return an empty list
    return results['documents'][0] if results.get('documents') else []

def make_prompt(query: str, relevant_passage: str) -> str:
    """Create a prompt combining the query with relevant passages."""
    return f"Question: {query}\n\nContext:\n{relevant_passage}"

def convert_passages_to_context(passages: List[str]) -> str:
    """Convert a list of passages into a single context string."""
    return "\n".join(passages)

# --- MAIN APP ---
def main():
    print("Choose an option:")
    print("1 - Enter a URL and ask questions about its content")
    print("2 - Generate a blog based on a topic")
    print("3 - Ask questions from a PDF document (after page 7)")

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        url = input("Enter URL to scrape: ").strip()
        content = scrape_url(url)
        print("\nScraping Done!\n")

        while True:
            user_query = input("Ask a question (or type 'exit' to quit): ").strip()
            if user_query.lower() == "exit":
                break
            answer = generate_response(user_query, content)
            print("\nAI Answer:", answer)

    elif choice == "2":
        topic = input("Enter a topic for the blog: ").strip()
        if not topic:
            logging.error("No topic provided.")
            sys.exit(1)
        search_results = search_duckduckgo(topic)
        blog = generate_blog(topic, search_results)
        if blog:
            print("\n--- Generated Blog ---\n", blog)
            save_path = save_blog(blog, topic)
            print(f"\nBlog saved to: {save_path}")
        else:
            print("Failed to generate blog.")

    elif choice == "3":
        file_path = input("Enter full path to PDF file: ").strip()
        question = input("Enter your question about the PDF: ").strip()

        raw_text = extract_text_from_pdf(file_path)
        cleaned = clean_extracted_text(raw_text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
        docs = [doc.page_content for doc in splitter.create_documents([cleaned])]

        db = create_chroma_db(docs, "sme_db")
        passages = get_relevant_passages(question, db, n_results=5)
        context = convert_passages_to_context(passages)
        prompt = make_prompt(question, context)

        print("\nAI Answer:\n")
        print(generate_response(question, context))

    else:
        print("Invalid choice. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




