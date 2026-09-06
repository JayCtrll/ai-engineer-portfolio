FROM python:3.10-slim

WORKDIR /code

# 安装系统依赖（如需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY ./app /code/app
COPY ./data /code/data
COPY .env /code/.env   # 注意：通常不推荐将 .env 提交到镜像，但演示可以

# 设置环境变量（可选）
ENV PYTHONUNBUFFERED=1

# 暴露端口（FastAPI 8000 和 Gradio 7860）
EXPOSE 8000 7860

# 默认启动 FastAPI 服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]