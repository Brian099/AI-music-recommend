# 使用官方轻量 Python 运行环境作为基础镜像
FROM python:3.12-slim

# 安装系统依赖：ffmpeg 是 librosa 解码音频文件（mp3/flac等）所必需的底座
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 优先拷贝依赖项文件，方便 Docker 缓存
COPY requirements.txt .

# 安装 Python 项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝代码和模型权重文件
COPY . .

# 声明容器对外暴露 8000 端口
EXPOSE 8000

# 设置默认的环境变量
ENV QDRANT_URL=http://qdrant:6333
ENV MUSIC_DIR=/music
ENV PYTHONIOENCODING=utf-8

# 启动 Web 服务
CMD ["uvicorn", "infer.app:app", "--host", "0.0.0.0", "--port", "8000"]
