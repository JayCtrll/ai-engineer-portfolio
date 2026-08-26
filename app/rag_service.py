# app/rag_service.py
import os
from langchain_core.documents import Document
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from sentence_transformers import CrossEncoder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

CHROMA_DIR = "./chroma_db"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

store = {}

logger = logging.getLogger(__name__)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        logger.info(f"[Session] Create new session: {session_id}")
    else:
        logger.debug(f"[Session] Reuse existing session: {session_id}")
    return store[session_id]

def rerank_documents(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """
    使用CrossEncoder对文档进行重排序
    :param query: 用户查询
    :param docs: 文档列表，每个文档是一个langchain Document对象
    :param top_k: 返回的文档数量
    :return: 重排序后的文档列表
    """
    logger.debug(f"[Rerank] input docs count={len(docs)}, top_k={top_k}")
    if not docs:
        logger.warning("[Rerank] input docs is empty")
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    result_docs = [doc for doc, score in scored_docs[:top_k]]
    logger.debug(f"[Rerank] output docs count={len(result_docs)}")
    return result_docs

def build_vectorstore_from_directory(directory: str = "data/knowledge_base"):
    logger.info(f"[VectorStore] build from directory: {directory}")
    loaders = [
        DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    logger.info(f"[VectorStore] build complete, chunk size:{len(chunks)}")
    return vectorstore

def load_vectorstore():
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
    else:
        return build_vectorstore_from_directory()
    
def load_or_create_vectorstore():
    """
    加载磁盘持久化Chroma向量库，返回检索器retriever
    persist_directory="./chroma_db"
    """
    logger.info("[VectorStore] load or create chroma db")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    logger.info("[VectorStore] retriever ready, k=6")
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
Answer the user's question strictly based on the provided context and chat history.
If the answer cannot be found in the context, reply exactly with 'I don't know'.
Do NOT invent or add any knowledge outside the given context.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:
"""
_prompt = PromptTemplate(
    template=_prompt_str,
    input_variables=["context", "question", "chat_history"]
)

# query rewrite prompt
_query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert follow‑up question into a self‑contained search query. Output ONLY the query text, nothing else."),
    ("human", "Chat history:\n{chat_history}\n\nLatest question: {question}")
])
_query_rewrite_chain = _query_rewrite_prompt | _llm | StrOutputParser()


# 自定义Runnable：实现【检索top6 → rerank取top3 → 格式化context】
def retrieval_and_rerank(inputs: dict) -> dict:
    original_question = inputs["question"]
    chat_history = inputs["chat_history"]
    history_str = get_buffer_string(chat_history)

    rewritten_query = _query_rewrite_chain.invoke({
        "chat_history": history_str,
        "question": original_question
    })

    # 增强清洗：去除各种模型输出的垃圾前缀
    garbage_prefix = ["Rewritten question:", "Rewritten standalone query:", "AI:", "User:", "Human:"]
    for prefix in garbage_prefix:
        rewritten_query = rewritten_query.replace(prefix, "")
    rewritten_query = rewritten_query.replace('"', '').replace("'", "").strip()

    # 兜底降级：如果改写后query太短，直接使用原始问题
    if len(rewritten_query) <= 8:
        rewritten_query = original_question

    logger.info(f"[Retrieval] original_q='{original_question}' | rewritten_q='{rewritten_query}'")

    raw_docs = _retriever.invoke(rewritten_query)
    logger.info(f"[Retrieval] retrieved doc count={len(raw_docs)}")

    reranked = rerank_documents(rewritten_query, raw_docs, top_k=3)
    context_text = _format_docs(reranked)
    logger.debug(f"[Retrieval] context snippet: {context_text[:200]}")

    return {
        "question": original_question,
        "context": context_text,
        "chat_history": chat_history
    }

# 构建RAG链（全局只构建一次）
# _rag_chain = (
#     {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
#     | _prompt
#     | _llm
#     | StrOutputParser()
# )

# 组装基础链
_base_rag_chain = (
    RunnableLambda(retrieval_and_rerank)
    | _prompt
    | _llm
    | StrOutputParser()
)

# 包装为支持会话历史的链
_rag_chain_with_history = RunnableWithMessageHistory(
    _base_rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def answer_question(query: str, session_id: str) -> str:
    """
    多轮RAG问答入口函数，供FastAPI直接调用
    流程：向量库检索 top_k=6 → CrossEncoder重排序选出前3 → LLM结合历史生成答案
    :param query: 用户提问字符串
    :param session_id: 会话ID，用于区分不同用户对话，内存保存历史
    :return: RAG返回的答案文本
    """
    logger.info(f"[answer_question] session_id={session_id}, query='{query}'")
    resp = _rag_chain_with_history.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}}
    )
    logger.info(f"[answer_question] session_id={session_id}, answer preview: {resp[:120]}")
    return resp


# 本地调试入口（直接运行本文件测试多轮）
if __name__ == "__main__":
    sid = "test-session-001"
    ans1 = answer_question("What is the main topic of the document?", session_id=sid)
    print(f"Q1: What is the main topic of the document?\nA1: {ans1}\n")

    ans2 = answer_question("Please tell me more details", session_id=sid)
    print(f"Q2: Please tell me more details\nA2: {ans2}")