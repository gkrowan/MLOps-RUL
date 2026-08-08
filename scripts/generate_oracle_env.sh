#!/usr/bin/env bash
set -euo pipefail

if [[ -e .env ]]; then
  echo ".env already exists; refusing to overwrite it." >&2
  exit 1
fi

command -v openssl >/dev/null || {
  echo "openssl is required to generate secrets." >&2
  exit 1
}

random_secret() {
  openssl rand -hex 24
}

cat > .env <<EOF
BIND_ADDRESS=127.0.0.1
POSTGRES_USER=mlops
POSTGRES_PASSWORD=$(random_secret)
AIRFLOW_SECRET_KEY=$(random_secret)
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=$(random_secret)
MLFLOW_ARTIFACT_BUCKET=mlflow
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=$(random_secret)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(random_secret)
EOF

chmod 600 .env
echo "Created .env with random secrets. Keep it private and back it up securely."
