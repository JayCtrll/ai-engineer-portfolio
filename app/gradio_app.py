import gradio as gr
import uuid
from rag_service import answer_question, build_vectorstore_from_directory


def ask(question, history, session_id):
    # 传入当前会话id，支持RAG多轮对话历史
    answer = answer_question(question, session_id=session_id)
    return answer


def rebuild_index():
    build_vectorstore_from_directory()
    return "向量索引已重建！"


with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# 📚 RAG 知识库问答助手")
    # 每个浏览器会话独立保存session_id
    session_id_state = gr.State()

    # 页面加载时生成唯一session_id
    demo.load(lambda: uuid.uuid4().hex, outputs=session_id_state)

    with gr.Tab("问答"):
        chatbot = gr.ChatInterface(
            fn=ask,
            additional_inputs=[session_id_state]
        )
    with gr.Tab("管理"):
        rebuild_btn = gr.Button("重建向量索引")
        rebuild_output = gr.Textbox(label="状态")
        rebuild_btn.click(rebuild_index, outputs=rebuild_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
