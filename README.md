# VoiceLab

An internal benchmarking platform for evaluating Automatic Speech Recognition (ASR) models.

VoiceLab provides a standardized workflow for benchmarking Whisper models against Mozilla Common Voice, tracking experiments with MLflow, and comparing results through a dashboard.

Designed to be extensible. New ASR models can be added by implementing a new transcriber in models.py, and new datasets can be integrated by extending dataset_manager.py while preserving the common benchmark sample interface.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- A [Mozilla Data Collective](https://mozilladatacollective.com) account and API key

### 1. Clone and install

```bash
git clone https://github.com/your-org/voice-lab.git
cd voice-lab
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Mozilla Data Collective API key:

```
MDC_API_KEY=your_api_key_here
```

> **First time only:** Visit the [dataset page](https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5) and accept the terms before running a benchmark. The dataset (~34MB) downloads automatically on the first run.

### 3. Start the services

```bash
# Terminal 1: API server
uv run uvicorn voicelab.api.main:app --reload

# Terminal 2: Dashboard
uv run streamlit run dashboard/app.py
```

- **API:** http://localhost:8000
- **Dashboard:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

### 4. Run a benchmark

Open the dashboard at http://localhost:8501, select a model, and click **Run Benchmark**.

---

## Docker Setup

Docker runs both services with a single command. Data is persisted in Docker volumes.

> **Note:** Docker on macOS cannot access Apple's GPU (MPS). Benchmarks run on CPU inside Docker. For MPS acceleration, use the native setup above.

### Build and run

```bash
docker compose up --build
```

- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000

### Stop

```bash
docker compose down
```

Benchmark results and dataset cache are preserved in Docker volumes (`voicelab-data`, `voicelab-mlruns`, `voicelab-mlflow-db`). To remove everything including data:

```bash
docker compose down -v
```

### Configuration

Create a `.env` file with your API key before running `docker compose up`. The file is mounted into the API container.

---

## Project Structure

```
voice-lab/
├── voicelab/                    # Backend Python package
│   ├── config.py                # Configuration loading
│   ├── device.py                # MPS/CPU device selection
│   ├── dataset_manager.py       # Dataset download, caching, subset selection
│   ├── models.py                # Whisper model registry and transcriber
│   ├── evaluation.py            # WER, CER, latency metrics
│   ├── tracking.py              # MLflow experiment tracking
│   ├── benchmark_pipeline.py    # Benchmark orchestration
│   ├── errors.py                # Exception hierarchy
│   └── api/
│       ├── main.py              # FastAPI routes
│       └── schemas.py           # Request/response models
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   └── api_client.py            # API client wrapper
├── configs/
│   └── default.yaml             # Default benchmark settings
├── tests/                       # Test suite
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Models

| Name | HF Model ID | Parameters |
|------|-------------|------------|
| whisper-tiny | openai/whisper-tiny | 39M |
| whisper-base | openai/whisper-base | 74M |
| whisper-small | openai/whisper-small | 244M |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| WER | Word Error Rate (corpus-level) |
| CER | Character Error Rate (corpus-level) |
| Avg Latency | Mean per-sample inference time |
| RTF | Real-Time Factor (inference time / audio duration) |
| Total Runtime | End-to-end benchmark duration |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + device info |
| GET | `/models` | List available models |
| GET | `/datasets` | Dataset metadata |
| POST | `/benchmarks` | Run a benchmark (synchronous) |
| GET | `/runs` | List all runs |
| GET | `/runs/{id}` | Run details |
| GET | `/runs/{id}/predictions` | Per-sample predictions |
| GET | `/runs/{id}/export.csv` | Download predictions CSV |
| GET | `/runs/compare?ids=a,b` | Compare runs |
| GET | `/runs/export.csv?ids=a,b` | Download comparison CSV |

## Tests

```bash
uv sync --group dev
uv run pytest tests/ -v
```

---

## Dataset

**Common Voice Scripted Speech 26.0 — Southern American English**

- Source: [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5)
- License: CC0-1.0
- Format: MP3 audio + TSV metadata
- Benchmark subset: 200 clips (configurable), selected with a fixed random seed for reproducibility
