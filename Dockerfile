FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/vertex_ai.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    google-cloud-storage \
    pyyaml

COPY src/ /app/src/
COPY setup.py /app/

RUN pip install -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.training.train"]
