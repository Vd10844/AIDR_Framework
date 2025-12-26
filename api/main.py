import sys
import os
from pathlib import Path

# Fix Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict
from src.core.pipeline import DocumentPipeline

app = FastAPI(title="AIDR Document Extraction", version="1.0.0")

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Starting AIDR API...")
    pipeline = DocumentPipeline("config/config.yaml")
    print("Pipeline ready!")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "PP-StructureV2", "version": "1.0.0"}

@app.post("/extract")
async def extract_kvps(file: UploadFile = File(...)):
    """Extract lotno, model, elevation, garage_swing from order forms"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    content = await file.read()
    result = await pipeline.extract(content)
    return {
        "status": result.status,
        "kvps": result.kvps,
        "confidences": result.confidences,
        "hitl_required": any(c < 0.8 for c in result.confidences.values())
    }
