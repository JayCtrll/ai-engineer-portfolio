from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/sample.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:2]):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content[:200])