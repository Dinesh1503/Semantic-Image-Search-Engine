# Semantic Image Search Engine

A semantic image search system that enables natural language search over image collections. The system uses vision-language models to generate detailed captions for images and embedding models to vectorize both captions and user queries, enabling semantic similarity-based image retrieval.

## Features

- **Natural Language Search**: Search images using descriptive text queries
- **AI-Powered Captioning**: Automatically generates detailed captions for images using vision-language models
- **Semantic Similarity**: Uses vector embeddings for accurate semantic matching
- **Categorized Results**: Results classified as "Good matches" or "Other results" based on similarity thresholds
- **Modern UI**: Clean, dark-themed React frontend with responsive image grid

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│  FastAPI Server │────▶│  PostgreSQL +   │
│   (Vite)        │◀────│                 │◀────│  pgvector       │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐           ┌───────▼───────┐
              │  Caption  │           │   Embedding   │
              │   Model   │           │     Model     │
              │ (BLIP/VL) │           │ (Qwen3-0.6B)  │
              └───────────┘           └───────────────┘
```

### Workflow

**Image Indexing:**
1. Scan directories for images
2. Generate detailed captions using VLM (Vision-Language Model)
3. Convert captions to 1024-dimensional embeddings
4. Store in PostgreSQL with pgvector extension

**Search:**
1. User enters natural language query
2. Query is vectorized using the embedding model
3. Similarity search performed using pgvector
4. Results classified and returned to frontend

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Uvicorn |
| Frontend | React 19, Vite, Axios |
| Caption Models | BLIP, Qwen3-VL-4B |
| Embedding Model | Qwen3-Embedding-0.6B |
| Vector Database | PostgreSQL + pgvector |
| ML Frameworks | PyTorch, Transformers, MLX |
| Acceleration | Apple Metal (MPS) |


## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL with pgvector extension

### Backend Setup

```bash
# Navigate to project root
cd "Semantic Image Search Engine"

# Install Python dependencies
pip install -e .

# Or using uv
uv pip install -e .
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### Database Setup

1. Install PostgreSQL and pgvector extension
2. Create a database for the project
3. Configure connection in environment variables

## Usage

### Start the Backend

```bash
cd engine
uvicorn server:app --reload --port 8000
```

### Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`


## API Reference

### POST /search

Search for images using a natural language query.

**Request:**
```json
{
  "caption": "golden retriever running in a park"
}
```

**Response:**
```json
{
  "good": [[index, similarity, path, filename], ...],
  "bad": [[index, similarity, path, filename], ...],
  "irrelevant": [[index, similarity, path, filename], ...]
}
```

**Classification Thresholds:**
- `good`: similarity >= 0.4
- `bad`: 0.2 <= similarity < 0.4
- `irrelevant`: similarity < 0.2

### GET /images/{path}

Serves static image files.

## Models

### Caption Models

| Model | Framework | Description |
|-------|-----------|-------------|
| BLIP | PyTorch | `Salesforce/blip-image-captioning-large` - Fast, general-purpose captions |
| Qwen3-VL-4B | Transformers | `Qwen/Qwen3-VL-4B-Instruct` - Detailed, structured captions |
| MLX Qwen3-VL | MLX | 4-bit quantized version for Apple Silicon |

### Embedding Model

- **Qwen3-Embedding-0.6B**: Generates 1024-dimensional normalized vectors
- Uses different prompts for captions vs queries for optimal retrieval

## Configuration

Configuration lives in `engine/.env`, which is gitignored. Copy `engine/.env.example` and fill in real values:

| Variable | Purpose |
|----------|---------|
| `DB`, `DB_USER`, `PASSWORD`, `DB_HOST`, `PORT` | PostgreSQL connection details |
| `IMAGES_PATH` | Absolute path to the directory served under `/images`. The `engine/test_data/a` default is not committed, so set this or the server refuses to start |
| `MODELS_PATH` | Directory of locally downloaded model weights (default `engine/models`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default `http://localhost:5173`) |
| `API_KEY` | When set, `/search` requires a matching `X-API-Key` header |
| `HOST`, `SERVER_PORT` | Bind address for `python server.py` (default `127.0.0.1:8000`) |

Start the database with the same credentials:

```bash
docker compose --env-file engine/.env up -d
```

Never commit `engine/.env`; rotate any credential that has been committed.

## License

MIT
