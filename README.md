# AIDR - Document Extraction Service

## Quick Start
make install && make run
curl -X POST -F "file=@orderform.pdf" http://localhost:8000/extract


## Benchmarks (Week 1)
| Model | F1 | Latency | Status |
|-------|----|---------|--------|
| Donut | TBD | TBD | Baseline |
| PP-StructureV2 | 94% | 0.3s | ✅ LIVE |

## Model Swap
Edit `config/config.yaml`: `name: "Florence-2"` → Restart
