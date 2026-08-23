from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载并分割文档（复用周二逻辑）
loader = PyPDFLoader("data/sample.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# 1. 初始化嵌入模型 all‑MiniLM‑L6‑v2
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # 有gpu改成 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)

# 2. 存入Chroma，持久化目录 ./chroma_db
persist_dir = "./chroma_db"
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_dir
)
# 重新加载向量数据库，不用再次向量化
# db = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedding
# )

# 3. 相似度搜索 query: What is the main topic? 返回前3条
query_text = "What is the main topic?"
docs = db.similarity_search(query_text, k=3)

# 打印结果
for idx, doc in enumerate(docs):
    print(f"===== Result {idx+1} =====")
    print("Content:", doc.page_content[:200])
    print("Metadata:", doc.metadata)
    print()
# docs_with_score = db.similarity_search_with_score(query_text, k=3)
# for doc, score in docs_with_score:
#     print("score:", score, "content:", doc.page_content[:200])