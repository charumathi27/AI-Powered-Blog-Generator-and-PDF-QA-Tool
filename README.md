AI-Powered Blog Generator and PDF QA Tool
Description:
Developed an end-to-end AI application that integrates web scraping, DuckDuckGo search, Gemini LLM, and ChromaDB for automated blog generation and document question answering.

Key Contributions:

Built a blog generation module that:

Scrapes websites using BeautifulSoup.

Performs web search with DuckDuckGo.

Generates engaging, structured blogs using Gemini generative AI models.

Saves outputs with logging and automated filename management.

Developed a PDF Question Answering system:

Extracts and cleans text from PDFs using PyPDF2.

Splits and embeds text chunks into ChromaDB vector database.

Retrieves relevant passages for user queries and generates context-based answers via Gemini.

Designed a modular, menu-driven CLI application with functionalities for:

URL-based QA.

Blog generation from topic search.

PDF document QA after preprocessing.

Technologies Used: Python, Gemini AI (Google Generative AI), DuckDuckGo Search API, BeautifulSoup, ChromaDB, LangChain, PyPDF2, Logging, TQDM

Impact: Automated blog writing and document analysis workflows, improving research productivity and content generation efficiency.
