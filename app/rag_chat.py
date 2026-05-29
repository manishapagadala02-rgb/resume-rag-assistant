from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load PDF
loader = PyPDFLoader("uploads/Developer__AI_ML.pdf")
documents = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vectorstore"
)

# User question
query = input("Ask a question about the resume: ")

# Retrieve relevant chunks
results = db.similarity_search(query, k=3)

# Combine retrieved chunks
context = "\n\n".join([doc.page_content for doc in results])

# Create prompt
prompt = f"""
You are an intelligent AI Resume ATS Assistant.

Use ONLY the resume context below.

If information is missing, say:
"Information not found in the resume."

Resume Context:
{context}

Question:
{query}

Professional Answer:
"""

# Load local LLM
llm = OllamaLLM(model="gemma:2b")

# Generate answer
response = llm.invoke(prompt)

# Print answer
print("\n🤖 AI Answer:\n")
print(response)