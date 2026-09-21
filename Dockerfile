FROM python:3.11-slim

RUN apt update && apt install -y --no-install-recommends \
    libsndfile1 \
    && apt clean

WORKDIR /app

COPY giga-am-wyoming.py .

RUN pip install --no-cache-dir wyoming numpy onnx-asr[cpu,hub]

EXPOSE 10500

ENTRYPOINT ["python3", "giga-am-wyoming.py"]
