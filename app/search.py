from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing vector database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

# Ask a question
query = "What cloud technologies does candidate know?"

# Search similar chunks
results = db.similarity_search(query)

# Print best matching chunk
print("\n🔍 Most Relevant Result:\n")
print(results[0].page_content)