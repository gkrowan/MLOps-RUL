# Oracle Cloud deployment (Ubuntu ARM64)

This profile targets the project's Ubuntu 24.04 ARM64 instance. All web UIs bind
to the instance loopback interface, so they are reachable only through SSH. Do
not add public ingress rules for ports 3000, 5000, 8080, or 9001.

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
  -L 9001:127.0.0.1:9001 `
  ubuntu@207.211.182.178
```

Then open Airflow at <http://localhost:8080>, MLflow at
<http://localhost:5000>, Grafana at <http://localhost:3000>, and MinIO at
<http://localhost:9001>. Credentials are stored in the server's `.env` file.

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
