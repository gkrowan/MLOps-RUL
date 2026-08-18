# FastAPI model serving

The `api` Compose service loads a registered MLflow model when the container
starts. By default it resolves the numerically latest version of
`femto-rul-model`. For production, set `API_MODEL_REFERENCE=champion` after a
model has been promoted to that registry alias.

## Configuration

```dotenv
API_MODEL_NAME=femto-rul-model
API_MODEL_REFERENCE=latest
API_KEY=replace-with-a-random-secret
```

The API reaches MLflow over the internal Compose network. MLflow proxies model
artifacts from MinIO, so the API does not need MinIO credentials.

## Endpoints

```text
GET  /health
GET  /model-info
POST /predict
GET  /docs
```

`/health` is public so Docker can monitor the service. When `API_KEY` is set,
`/model-info` and `/predict` require an `X-API-Key` header.

First inspect the exact feature names recorded in the model signature:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/model-info
```

Then send exactly those features:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "features": {
      "condition": 1,
      "rms_horiz": 0.42
    }
  }'
```

The shortened example above is illustrative; a real request must include every
feature returned by `/model-info` and no additional fields.

The model is loaded once at container startup. After registering or promoting a
new model, reload it with:

```bash
docker compose -f docker-compose.yml -f docker-compose.oracle.yml restart api
```

The OCI profile binds services according to `BIND_ADDRESS`. Keep it set to
`127.0.0.1` and access the API through an SSH/VPN tunnel unless an authenticated
TLS reverse proxy protects the service.
