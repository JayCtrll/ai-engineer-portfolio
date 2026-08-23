from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv() # 自动读取项目下.env文件

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    return f"The weather in {city} is sunny, 30°C"

llm = ChatOllama(
    model="qwen3:4b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0,       # 关闭随机性，防止幻觉
)

# 将工具绑定到 LLM，开启 function calling
llm_with_tools = llm.bind_tools([get_weather])

# ---------------------- 3. 用户提问：What is the weather in Singapore? ----------------------
user_query = "What is the weather in Singapore?"
messages = [HumanMessage(content=user_query)]

# 第一轮：模型输出工具调用请求，不直接出答案
ai_msg = llm_with_tools.invoke(messages)
print("===模型输出（工具调用请求）===")
print(ai_msg)

# 把模型返回的工具调用追加到消息列表
messages.append(ai_msg)

# ----------------------4. 执行工具调用（手动执行工具）----------------------
# langchain 标准做法：解析 ai_msg.tool_calls，执行对应函数
for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_weather": get_weather}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    print(f"\n===工具返回结果===")
    print(tool_output)
    # 将工具结果封装成消息加入会话
    from langchain_core.messages import ToolMessage
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

# ----------------------5. 将工具结果传回模型，生成最终自然语言回答 ----------------------
final_response = llm_with_tools.invoke(messages)
print("\n====最终回答====")
print(final_response.content)