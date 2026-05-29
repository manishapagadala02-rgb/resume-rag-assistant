from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("uploads/Developer__AI_ML.pdf")

documents = loader.load()

print(documents[0].page_content)