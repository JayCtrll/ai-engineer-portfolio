from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------- 1.加载Embedding与持久化Chroma向量库 ----------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# 读取磁盘上的向量库
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)
# 构建检索器，返回top‑3文档片段
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ---------------------- 2.初始化本地Ollama llama3‑8b ----------------------
llm = OllamaLLM(
    model="qwen2:1.5b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0,       # 关闭随机性，防止幻觉
    num_ctx=8192,          # 设置上下文窗口
    timeout=120
)

# ----------------------3.自定义Prompt模板：无信息输出 I don't know ----------------------
prompt_str = """
Answer the user's question strictly based on the provided context.
If the answer cannot be found in the context, reply exactly with 'I don't know'.
Do NOT invent or add any knowledge outside the given context.

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_str, input_variables=["context", "question"])

# 4.拼接检索文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5.构建LCEL RAG链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6.执行查询
answer = rag_chain.invoke("What is the main topic of the document?")
print(answer)