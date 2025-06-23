# SensitiveScan

SensitiveScan is a FastAPI-based service for scanning uploaded files for sensitive information using [Microsoft Presidio](https://github.com/microsoft/presidio). The project includes a web API, load testing with Locust, monitoring with Prometheus and Grafana, and centralized logging with Loki and Promtail.

## Features

- **/scan endpoint**: Upload a file and receive detected sensitive data entities.
- **Presidio Analyzer**: Uses Microsoft Presidio for PII detection.
- **NGINX Proxy**: Reverse proxy for the API.
- **Load Testing**: Locust integration for performance testing.
- **Monitoring**: Prometheus and Grafana dashboards.
- **Centralized Logging**: Loki and Promtail for log aggregation.

## Project Structure

```
.
├── app.py                # FastAPI application
├── test_app.py           # Unit tests for the API
├── locustfile.py         # Locust load test script
├── test_scan.html        # Simple HTML client for manual testing
├── Dockerfile            # Dockerfile for the API
├── Dockerfile.locust     # Dockerfile for Locust
├── docker-compose.yml    # Multi-service orchestration
├── nginx.conf            # NGINX configuration
├── prometheus.yml        # Prometheus configuration
├── promtail-config.yaml  # Promtail configuration
├── logging.yaml          # Python logging configuration
├── requirements.txt      # Python dependencies
├── deployment.yaml       # Kubernetes HPA example
└── logs/                 # Log output directory
```

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Build and Run

1. **Clone the repository**  
   ```sh
   git clone <your-repo-url>
   cd SensitiveScan
   ```

2. **Start all services**  
   ```sh
   docker compose up --build
   ```

3. **Access the services:**
   - FastAPI: [http://localhost:8100/scan](http://localhost:8100/scan)
   - Presidio Analyzer: [http://localhost:5100](http://localhost:5100)
   - NGINX Proxy: [http://localhost:8080/scan](http://localhost:8080/scan)
   - Locust UI: [http://localhost:8090](http://localhost:8090)
   - Prometheus: [http://localhost:9090](http://localhost:9090)
   - Grafana: [http://localhost:9091](http://localhost:9091)
   - Loki API: [http://localhost:3100](http://localhost:3100)

### API Usage

#### `/scan` Endpoint

- **Method:** `POST`
- **Form field:** `file` (multipart/form-data)
- **Response:** JSON with detected entities

Example using `curl`:
```sh
curl -F "file=@yourfile.txt" http://localhost:8100/scan
```

### Manual Testing

Open [test_scan.html](test_scan.html) in your browser to upload a file and see the API response.

### Running Tests

```sh
pytest test_app.py
```

### Load Testing

1. Place a sample file named `sample.txt` in the project root.
2. Access the Locust UI at [http://localhost:8090](http://localhost:8090) and start a test.

### Monitoring & Logs

- **Prometheus** scrapes metrics from the API and Locust.
- **Grafana** dashboards can be configured at [http://localhost:9091](http://localhost:9091).
- **Logs** are collected from `/logs/app/server.log` and sent to Loki via Promtail.

## 🔐 Authentication

SensitiveScan requires an **API token** for scanning. Tokens are stored and managed via the `/tokens` endpoint.

### Managing Tokens

* **Create Token:**

  ```bash
  curl -X POST http://localhost:8100/tokens \
    -H "Content-Type: application/json" \
    -d '{"description": "My test token", "days_valid": 90}'
  ```

* **Use Token in Requests:**
  Add `x-api-token` in the request header:

  ```bash
  curl -F "file=@yourfile.txt" \
    -H "x-api-token: YOUR_TOKEN_HERE" \
    http://localhost:8100/scan
  ```

* **Get All Tokens:**

  ```bash
  curl http://localhost:8100/tokens
  ```

* **Deactivate or Delete Tokens:**

  ```bash
  curl -X PATCH http://localhost:8100/tokens/1?is_active=0
  curl -X DELETE http://localhost:8100/tokens/1
  ```

---

## 📥 `/scan` Endpoint Enhancements

The `/scan` endpoint now returns:

* `file`: Filename of the uploaded file.
* `sensitive`: Boolean, `true` if sensitive data was found.
* `findings`: A list of detected sensitive data entities (up to 10).
* `embedding`: 768-dimension vector of the document’s semantic content.

### Example JSON Response

```json
{
  "file": "document.pdf",
  "sensitive": true,
  "findings": [
    {
      "entity_type": "PHONE_NUMBER",
      "start": 10,
      "end": 22,
      "score": 0.85,
      "text": "123-456-7890"
    }
  ],
  "embedding": [0.021, -0.018, ..., 0.004]  // vector of 768 values
}
```

---

## 🧠 Embedding Model

SensitiveScan uses the [FlagEmbedding](https://huggingface.co/BAAI/bge-base-en) model for converting file content into semantic vectors.

* **Model:** `BAAI/bge-base-en`
* **Framework:** PyTorch
* **Device:** Automatically uses CUDA if available

These embeddings can be used for similarity search, clustering, or visualization.

---

## 🧪 Local Development Tips

* Run with multiple API containers:

  ```bash
  docker-compose up --scale app=3 --build
  ```

* Shut everything down and remove volumes:

  ```bash
  docker-compose down -v
  ```

🔐 Token Management
Method	Endpoint	Description
GET	/tokens	List all API tokens
POST	/tokens	Create a new API token
PATCH	/tokens/{token_id}	Toggle token active status
DELETE	/tokens/{token_id}	Delete an API token

📥 File Scanning
Method	Endpoint	Description
POST	/scan	Upload a file, detect sensitive data, embed, log
POST	/scan/sensitive	Upload a file, return only { sensitive: true/false }
GET	/scan	List all previous scans with filename, findings, embeddings

🔍 Semantic Search
Method	Endpoint	Description
GET	/search?query=...	Return top-k most semantically similar scanned files

## Configuration

- **Logging:** See [logging.yaml](logging.yaml)
- **Prometheus:** See [prometheus.yml](prometheus.yml)
- **Promtail:** See [promtail-config.yaml](promtail-config.yaml)
- **NGINX:** See [nginx.conf](nginx.conf)

## Kubernetes

A sample HorizontalPodAutoscaler is provided in [deployment.yaml](deployment.yaml).

## License

MIT License. See [LICENSE](LICENSE) if present.

---

*Powered by FastAPI, Presidio, and the Grafana observability stack.*

GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 12215' git push git@192.168.1.71:/sg_it/SensitiveScan

FastAPI	http://localhost:8100/scan
Presidio	http://localhost:5100
NGINX Proxy	http://localhost:8080/scan
Locust UI	http://localhost:8090
Prometheus	http://localhost:9090
Grafana	http://localhost:9091
Loki API	http://localhost:3100

docker-compose up --scale app=3 --build
docker-compose down -v