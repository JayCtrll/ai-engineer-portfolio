import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
# 加载.env环境变量
load_dotenv()

# 读取环境变量
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

llm = OllamaLLM(model="llama3:8b", base_url=ollama_base_url,api_key=ollama_api_key)

result = llm.invoke("What is edge AI?")
print(result)
