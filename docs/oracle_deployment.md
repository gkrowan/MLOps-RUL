# Oracle Cloud deployment (Ubuntu ARM64)

This profile targets the project's Ubuntu 24.04 ARM64 instance. All web UIs bind
to the instance loopback interface, so they are reachable only through SSH. Do
not add public ingress rules for ports 3000, 5000, 8080, 9000, or 9001.

## 1. Prepare the repository

```bash
sudo apt update
sudo apt install -y git openssl
git clone https://github.com/gkrowan/MLOps-RUL.git
cd MLOps-RUL
git checkout chris-eda
chmod +x scripts/generate_oracle_env.sh
./scripts/generate_oracle_env.sh
```

Record the generated passwords from `.env` in a password manager. Never commit
that file or paste it into an issue, chat, or notebook.

The generated `AIRFLOW_UID` makes the Airflow containers write bind-mounted logs
as the Ubuntu user instead of container root.

## 2. Start the services

```bash
docker compose -f docker-compose.yml -f docker-compose.oracle.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.oracle.yml ps
```

The named Docker volumes persist PostgreSQL, MLflow metadata, MinIO artifacts,
and Grafana state across container replacement and host reboots. They still live
on the boot volume, so configure OCI boot-volume backups separately.

## 3. Open an SSH tunnel

Run this on the Windows workstation and leave the terminal open:

```powershell
ssh -N `
  -L 8080:127.0.0.1:8080 `
  -L 5000:127.0.0.1:5000 `
  -L 3000:127.0.0.1:3000 `
  -L 9000:127.0.0.1:9000 `
  -L 9001:127.0.0.1:9001 `
  ubuntu@207.211.182.178
```

Then open Airflow at <http://localhost:8080>, MLflow at
<http://localhost:5000>, Grafana at <http://localhost:3000>, and the MinIO console at
<http://localhost:9001>. DVC and other S3-compatible clients use the MinIO API at
<http://localhost:9000>. Credentials are stored in the server's `.env` file.

## 4. Verify and operate

```bash
docker compose -f docker-compose.yml -f docker-compose.oracle.yml ps
docker compose -f docker-compose.yml -f docker-compose.oracle.yml logs --tail=100
docker compose -f docker-compose.yml -f docker-compose.oracle.yml pull
docker compose -f docker-compose.yml -f docker-compose.oracle.yml up --build -d
```

To stop containers without deleting data:

```bash
docker compose -f docker-compose.yml -f docker-compose.oracle.yml down
```

Do not add `-v` unless permanently deleting all local databases and artifacts is
intentional.

## Recover from an Airflow init permission failure

If the deployment was first started without `AIRFLOW_UID`, keep the existing
`.env` secrets and add the host UID:

```bash
cd ~/MLOps-RUL
grep -q '^AIRFLOW_UID=' .env || echo "AIRFLOW_UID=$(id -u)" >> .env
mkdir -p airflow/logs artifacts
sudo chown -R "$(id -u):$(id -g)" airflow/logs artifacts
docker compose -f docker-compose.yml -f docker-compose.oracle.yml down
docker compose -f docker-compose.yml -f docker-compose.oracle.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.oracle.yml ps
```

This does not remove named volumes. PostgreSQL, MLflow, MinIO, and Grafana data
remain intact.
