# AI Portfolio API — RAG 完整运行环境镜像
# 本地依赖与验证环境: fastapi-week1 (Python 3.10.12)
# 基础镜像用 Debian 12 (bookworm) slim: trixie(13) 已移除 openjdk-17，
# 而 pyspark 3.5 需要 Java 8/11/17（不支持 21）
FROM python:3.10-slim-bookworm

WORKDIR /code

# 国内网络环境下切换到清华 Debian 镜像源，避免 apt 访问官方源超时
# 兼容 deb822（/etc/apt/sources.list.d/debian.sources）与旧式（/etc/apt/sources.list）两种格式
RUN sed -i 's|http://deb.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# 系统依赖:
#   build-essential     编译部分 Python 包
#   libgomp1            torch CPU 版运行库
#   openjdk-17-jre      pyspark 运行需要 JVM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 国内网络环境：pip 走清华源加速（torch 单独用 --index-url 指定 pytorch CPU 源，不受影响）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 先装 CPU 版 torch：PyPI 默认 wheel 带 CUDA，体积巨大且容器内无用
# --no-deps: pytorch 官方 CPU index 不完整（缺 typing_extensions 等 wheel 与 flit_core 构建依赖），
#            仅拉取 torch 本体；其依赖在下一步 requirements.txt 安装时由清华源补齐
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu --no-deps

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 RAG 模型到镜像缓存（构建期下载，避免每次启动联网）
# all-MiniLM-L6-v2: 嵌入模型；ms-marco-MiniLM-L-6-v2: 重排序模型
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 复制项目文件（.env 不提交到镜像；配置通过 compose environment / 代码默认值提供）
COPY ./app /code/app
COPY ./data /code/data

# 设置环境变量（可选）
ENV PYTHONUNBUFFERED=1

# 暴露端口（FastAPI 8000 和 Gradio 7860）
EXPOSE 8000 7860

# 默认启动 FastAPI 服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
