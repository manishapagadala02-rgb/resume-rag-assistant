from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("uploads/Developer__AI_ML.pdf")

documents = loader.load()

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split into chunks
chunks = splitter.split_documents(documents)

# Print first chunk
print(chunks[0].page_content)