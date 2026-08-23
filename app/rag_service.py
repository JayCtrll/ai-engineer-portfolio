# app/rag_service.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def load_or_create_vectorstore():
    """
    加载磁盘持久化Chroma向量库，返回检索器retriever
    persist_directory="./chroma_db"
    """
    # 初始化Embedding
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # 加载本地持久化向量库
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    # 构建检索器 top‑3
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return retriever


def _format_docs(docs):
    """内部工具函数：把文档列表拼接成上下文字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


# 全局初始化：只执行一次（服务启动时加载，避免每次调用重复加载embedding/向量库）
_retriever = load_or_create_vectorstore()

# LLM初始化
_llm = OllamaLLM(
    model="qwen2:1.5b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0,
    num_ctx=8192,
    timeout=120
)

# Prompt模板
_prompt_str = """
Answer the user's question strictly based on the provided context.
If the answer cannot be found in the context, reply exactly with 'I don't know'.
Do NOT invent or add any knowledge outside the given context.

Context:
{context}

Question: {question}
Answer:
"""
_prompt = PromptTemplate(template=_prompt_str, input_variables=["context", "question"])

# 构建RAG链（全局只构建一次）
_rag_chain = (
    {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
    | _prompt
    | _llm
    | StrOutputParser()
)


def answer_question(query: str) -> str:
    """
    RAG问答入口函数，供FastAPI直接调用
    :param query: 用户提问字符串
    :return: RAG返回的答案文本
    """
    result = _rag_chain.invoke(query)
    return result


# 本地调试入口（直接运行本文件测试）
if __name__ == "__main__":
    ans = answer_question("What is the main topic of the document?")
    print(ans)