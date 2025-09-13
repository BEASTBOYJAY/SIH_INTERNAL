FROM nvidia/cuda:12.5.0-base-ubuntu22.04

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.10 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

COPY requirements.txt ./
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

COPY src/ ./src/
COPY app.py .
COPY id_detector.pt .
COPY yolov8x-face.pt .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]