# AIDR - AI-Powered Document Extraction Service

A production-ready FastAPI service for extracting key-value pairs from documents (PDFs and images) using advanced OCR and layout analysis models.

## Features

- **Multi-format Support**: Processes PDF documents and images
- **AI-Powered Extraction**: Uses PaddlePaddle PP-StructureV2 for accurate text and layout analysis
- **RESTful API**: Simple HTTP endpoints for document processing
- **Human-in-the-Loop (HITL)**: Automatic confidence scoring with manual review routing
- **Docker Deployment**: Containerized for easy deployment and scaling
- **Configurable Models**: Easy model swapping via YAML configuration
- **Audit Logging**: Comprehensive logging for compliance and debugging

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd AI_DR

# Build and run with Docker Compose
make deploy

# View logs
make logs

# Stop the service
make clean
```

The API will be available at `http://localhost:8000`.

### Local Development

```bash
# Install dependencies
make install

# Run the server
make run
```

## API Usage

### Extract Key-Value Pairs

```bash
curl -X POST \
  -F "file=@sample.pdf" \
  http://localhost:8000/extract
```

**Response:**
```json
{
  "result": {
    "garage_swing": "Yes",
    "lot_no": "123",
    "address": "123 Main St"
  },
  "confidence": {
    "garage_swing": 0.95,
    "lot_no": 0.98,
    "address": 0.89
  },
  "status": "approved"
}
```

### API Documentation

Access the interactive API docs at `http://localhost:8000/docs` when the server is running.

## Configuration

Edit `config/config.yaml` to change models or settings:

```yaml
model:
  name: "PP-StructureV2"  # or "Florence-2", "Donut"

pipeline:
  hitl_threshold: 0.8  # Confidence threshold for human review
```

## Model Performance

| Model | F1 Score | Latency | Status |
|-------|----------|---------|--------|
| PP-StructureV2 | 94% | 0.3s | ✅ Production |
| Donut | TBD | TBD | Baseline |
| Florence-2 | TBD | TBD | Experimental |

## Deployment

### Docker Compose

```bash
docker-compose up --build -d
```

### Environment Variables

- `DISABLE_MODEL_SOURCE_CHECK`: Set to `True` to skip model download checks

## Development

### Running Tests

```bash
make test
```

### Project Structure

```
├── api/                 # FastAPI application
├── src/                 # Core business logic
│   ├── core/           # Pipeline and interfaces
│   ├── models/         # AI model implementations
│   └── adapters/       # Storage adapters
├── config/             # Configuration files
├── data/               # Data directory
├── tests/              # Unit tests
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose setup
└── requirements.txt    # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[MIT License](LICENSE)

## Support

For issues and questions, please open a GitHub issue.
