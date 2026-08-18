FROM python:3.12.3-slim

WORKDIR /opt/project

COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY api /opt/project/api
COPY src /opt/project/src
ENV PYTHONPATH=/opt/project:/opt/project/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
