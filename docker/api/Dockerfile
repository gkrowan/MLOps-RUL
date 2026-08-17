FROM python:3.12.3-slim

WORKDIR /opt/project

COPY docker/api/requirements.txt ./docker/api/requirements.txt
RUN pip install --no-cache-dir -r docker/api/requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
