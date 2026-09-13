# 新闻分析系统（News RAG）

基于 **RSS 抓取 → PySpark 清洗 → MinIO 存储 → 向量化检索 → FastAPI RAG 问答** 的全链路新闻智能分析系统。每小时自动增量抓取最新新闻，支持中文语义问答，答案附带可溯源的新闻引用（标题/来源/链接/时间/情感/相关度）。

---

## 一、项目简介

本系统打通了新闻数据从采集到问答的完整链路：

- **数据管道**：定时抓取 BBC / Reuters / TechCrunch 的 RSS 源，经 PySpark 清洗（去重、去空、情感分析）后落入 MinIO 对象存储；
- **向量化**：读取 MinIO 中的清洗后新闻，用 `all-MiniLM-L6-v2` 嵌入为向量，存入 ChromaDB 向量库；
- **RAG 问答**：FastAPI 提供 `/news/ask` 端点，对用户提问执行相似性检索，交由 Ollama 本地大模型（qwen3:4b）生成带引用的答案；
- **自动更新**：每小时定时任务增量抓取，新新闻自动入库并重建向量库，无需人工干预。

适合作为 **AI 工程师作品集** 中展示数据工程 + 向量检索 + LLM 应用完整能力的项目。

---

## 二、架构图

```
                             +---------------------------+
                             |  news_refresh.py 每小时定时  |
                             |  增量抓取 -> 比对 -> 入库    |
                             +-------------+-------------+
                                           | 触发
                                           v
 +---------+   +---------+   +---------+   +----------+   +----------+
 |  RSS源   |-->| 新闻爬虫 |-->| JSON存储 |-->| PySpark  |-->|  MinIO   |
 |  BBC    |   |feedparser|   | data/   |   | 清洗/去重 |   | S3A 存储 |
 | Reuters |   |Beautiful |   | raw_news|   | 情感分析  |   | spark-   |
 |TechCrunch|  | Soup     |   | *.json  |   |          |   | bucket   |
 +---------+   +---------+   +---------+   +----------+   +----+-----+
                                                                 |
                                                     清洗后 parquet 读取
                                                                 |
                                                                 v
 +---------+   +------------+   +----------+   +------------+   +------+
 |  用户    |<--|  FastAPI   |<--| ChromaDB |<--|  向量化     |<--| RAG  |
 | 提问/答案 |  | /news/ask  |   | news_    |   | HuggingFace|   | 检索 |
 |  + 引用  |   |            |   | chroma_db|   | MiniLM-L6  |   | top_k|
 +---------+   +-----+------+   +----------+   +------------+   +------+
                     |
                     | Ollama (qwen3:4b) 生成答案
                     v
              +------------+
              |  本地 LLM   |
              +------------+
```

**数据流说明**

| 环节 | 模块 | 说明 |
|---|---|---|
| ① RSS 源 | `news_crawler.py` | feedparser 抓取 BBC/Reuters/TechCrunch，BeautifulSoup 清洗 HTML |
| ② JSON 存储 | `data/raw_news/*.json` | 抓取的原始新闻条目落盘为 JSON |
| ③ PySpark 清洗 | `spark_cleaner.py` / `news_refresh.py` | 去重、过滤空标题、关键词情感分析（positive/negative/neutral） |
| ④ MinIO | `s3a://spark-bucket/news/cleaned` | 清洗后 parquet 持久化（S3A 协议，path-style 访问） |
| ⑤ 向量化 | `news_rag.py` | `all-MiniLM-L6-v2` 嵌入 title+summary |
| ⑥ ChromaDB | `news_chroma_db/` | 向量持久化存储，相似性检索 |
| ⑦ FastAPI RAG | `main.py` `/news/ask` | 检索 top_k → Ollama 生成答案 → 返回引用列表 |

---

## 三、技术栈

| 类别 | 技术 | 用途 |
|---|---|---|
| 语言 | Python 3.10 | 全栈实现 |
| Web 框架 | FastAPI 0.141 + Uvicorn | REST API 服务 |
| 数据抓取 | feedparser 6.0 + BeautifulSoup4 | RSS 解析、HTML 清洗 |
| 大数据处理 | PySpark 3.5.9（Hadoop S3A） | 分布式清洗、情感分析 |
| 对象存储 | MinIO（Docker 容器 / 独立进程） | 清洗后新闻 parquet 持久化 |
| 向量化 | sentence-transformers 6.0（all-MiniLM-L6-v2） | 语义嵌入 |
| 向量库 | ChromaDB 1.5.9 | 向量存储与相似性检索 |
| LLM 编排 | LangChain 1.3（core/community/huggingface/ollama/chroma） | RAG 链路 |
| 本地大模型 | Ollama + qwen3:4b | 答案生成、中文问题翻译 |
| 部署 | Docker + Docker Compose（quay.io/minio/minio） | 一键容器化 |
| 定时任务 | cron（每小时） | 增量新闻检查 |

---

## 四、快速开始

### 环境依赖

| 组件 | 地址 | 默认凭据 |
|---|---|---|
| MinIO | `http://127.0.0.1:9000`（S3）/ `:9001`（控制台） | minioadmin / minioadmin |
| Ollama | `http://127.0.0.1:11434` | 模型 `qwen3:4b` |
| Spark jars | `/home/cjc/spark_jars/`（hadoop-aws 3.3.4 + aws-java-sdk-bundle） | - |

MinIO 配置支持环境变量覆盖：`MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`（默认值即上表）。

### 方式一：本地运行

```bash
# 1. 安装依赖（推荐 pyenv 虚拟环境，Python 3.10）
pip install -r requirements.txt

# 2. 一键数据管道：抓取 RSS -> 清洗 -> 写 MinIO -> 重建向量库
python app/news_refresh.py
# 输出示例：NEW_NEWS=0 total=93  或  NEW_NEWS=12 total=105

# 3.（可选）仅重建向量库（从 MinIO 现有数据）
python -c "from app.news_rag import rebuild_store; rebuild_store()"

# 4. 启动 API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. 测试新闻问答
curl -X POST http://127.0.0.1:8000/news/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "最近有哪些 AI 公司获得融资？", "top_k": 3}'
```

Swagger 文档：`http://127.0.0.1:8000/docs`

### 方式二：Docker Compose 部署

```bash
cd /home/cjc/projects/ai-engineer-portfolio
docker compose up -d --build
```

| 服务 | 镜像 | 端口 | 说明 |
|---|---|---|---|
| api | 本地构建（Dockerfile） | 8000 | FastAPI + RAG，挂载宿主机 `./news_chroma_db` |
| minio | `quay.io/minio/minio` | 9000 / 9001 | 对象存储，数据持久化到 `./minio_data` |

- 两服务接入 `news-net` 桥接网络，api 通过服务名 `minio:9000` 访问；
- **首次构建约 5-15 分钟**（CPU 版 torch + pyspark + 模型预下载，apt/pip 已切清华源）；
- 容器内 LLM 生成需访问宿主机 Ollama：在 compose 的 api 服务追加 `extra_hosts: ["host.docker.internal:host-gateway"]` 并设置 `OLLAMA_BASE_URL=http://host.docker.internal:11434`；
- 旧数据迁移：将原 MinIO 数据目录下的 `spark-bucket` 复制到 `./minio_data/` 后 `docker restart news_minio` 即可恢复。

### 定时增量更新

已配置 cron 定时任务「每小时新闻检查」，每小时执行：

```bash
cd /home/cjc/projects/ai-engineer-portfolio
/home/cjc/.pyenv/versions/fastapi-week1/bin/python app/news_refresh.py
```

输出约定：`NEW_NEWS=N total=M`（新增 N 条，库内共 M 条）/ `NEW_NEWS=0`（无新增）/ `ERROR: <原因>`。

---

## 五、API 文档

### 端点总览

| 方法 | 路径 | 说明 | 标签 |
|---|---|---|---|
| GET | `/health` | 健康探活 | System |
| GET/POST | `/echo` | 消息回声测试 | Echo |
| GET | `/secure-data` | Bearer Token 鉴权测试 | Auth |
| GET | `/sync-slow` / `/async-slow` | 同步/异步性能对比 | Performance |
| POST | `/upload` | PDF 上传并向量化（追加至 `chroma_db`） | RAG |
| POST | `/reset-db` | 清空 PDF 向量库 | RAG |
| POST | `/rebuild-index` | 重建文档向量索引 | RAG |
| POST | `/rag/ask` | 多轮文档 RAG 问答（session 会话） | RAG Chat |
| DELETE | `/rag/session/{session_id}` | 清除会话历史 | RAG Chat |
| **POST** | **`/news/ask`** | **新闻 RAG 问答（本系统核心）** | **News RAG** |

### POST /news/ask

基于新闻向量库的问答接口：相似性检索 top_k 条新闻 → Ollama 生成答案 → 返回答案与引用列表。

**请求体**

```json
{
  "query": "哪些公司被指控模型蒸馏？",
  "top_k": 3,
  "use_llm": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| query | string | ✅ | 用户提问（支持中文，自动翻译后检索） |
| top_k | int | ❌ | 引用新闻条数，默认 3，范围 1~10 |
| use_llm | bool | ❌ | 默认 true；false 时仅返回检索结果不调用 LLM |

**响应示例**

```json
{
  "query": "哪些公司被指控模型蒸馏？",
  "answer": "根据新闻库检索结果……",
  "sources": [
    {
      "title": "Nscale adds former OpenAI exec Fidji Simo to its board…",
      "link": "https://techcrunch.com/2026/09/11/…",
      "source": "TechCrunch",
      "published": "Fri, 11 Sep 2026 16:46:25 +0000",
      "sentiment": "neutral",
      "score": 1.7648
    }
  ]
}
```

**行为说明**
- 中文问题自动经 LLM 翻译为英文后检索（向量模型为英文语义空间），相关性低于阈值（`RELEVANCE_THRESHOLD=1.2`）时返回"未找到相关报道"；
- 无命中时不编造答案，`use_llm=false` 时跳过 LLM 调用；
- 引用中的 `score` 为检索相关度分，`sentiment` 为入库时情感分析结果（positive/negative/neutral）。

---

## 六、项目亮点

1. **全链路数据管道**：RSS → JSON → PySpark 清洗（去重/去空/情感分析）→ MinIO 持久化 → 向量化 → ChromaDB，数据自洽可溯源；
2. **中文问答支持**：自动检测中文问题并翻译为英文检索，配合相关性阈值过滤无关命中，解决英文向量模型的中文查询失配问题；
3. **情感分析随行**：清洗阶段即完成情感标注，回答中可展示新闻情绪倾向；
4. **答案可溯源**：每次回答附带引用新闻列表（标题/来源/链接/时间/情感/相关度），拒绝无依据编造；
5. **每小时自动更新**：定时任务增量抓取、去重入库、自动重建向量库，系统保持新鲜；
6. **部署友好**：Docker Compose 一键起 api + minio；向量库挂载卷复用，pyspark 懒加载保证无 Spark 环境也能运行问答服务；
7. **稳定构建**：Debian 12 基础镜像匹配 pyspark 的 Java 17 需求，apt/pip 清华源 + CPU 版 torch 适配国内网络与资源受限环境。
