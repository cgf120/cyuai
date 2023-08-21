FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
FROM python:3.10

WORKDIR /app


# 使用账号cgf120密码018797as克隆项目
RUN apt-get update && apt-get install -y git \
    && git clone https://github.com/cgf120/cyuai /app \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

# 启动Flask入口main.js
CMD python main.py