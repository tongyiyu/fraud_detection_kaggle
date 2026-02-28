# 改用Debian官方镜像（最稳定，国内可访问）
FROM debian:bullseye-slim

# 设置环境变量，避免Python输出缓冲、指定时区
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 更新源并安装基础依赖（Python 3.10 + 必要工具）
# 关键：所有命令在同一个RUN中，反斜杠必须放在行尾，且无多余空格/换行
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    gcc \
    g++ \
    libgomp1 \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 验证Python版本
RUN python --version && pip --version

# 设置工作目录
WORKDIR /app

# 复制requirements文件（先复制，利用Docker缓存）
COPY requirements.txt .

# 安装Python依赖（阿里云源+升级pip）
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动命令（添加超时配置，避免gunicorn卡死）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]