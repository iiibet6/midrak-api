FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /root/.camel_tools/data/morphology_db/calima-msa-r13

COPY resources/morphology/morphology.db \
     /root/.camel_tools/data/morphology_db/calima-msa-r13/morphology.db

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
