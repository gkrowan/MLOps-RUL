FROM apache/airflow:2.10.5-python3.12

COPY docker/airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=airflow:root src /opt/project/src
COPY --chown=airflow:root scripts /opt/project/scripts
ENV PYTHONPATH=/opt/project/src
