import sys
import os
from pathlib import Path
from loguru import logger

# Fix Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict
from src.core.pipeline import DocumentPipeline

app = FastAPI(title="AIDR Document Extraction", version="1.0.0")

pipeline = None


@app.get("/")
async def root():
    return {"message": "Welcome to AIDR Document Extraction API"}

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Starting AIDR API...")
    pipeline = DocumentPipeline("config/config.yaml")
    logger.info("Pipeline ready!")

@app.get("/health")
async def health_check():
    if pipeline is None:
        return {"status": "starting", "model": "TBD"}
    
    # Check if model has been initialized
    model_ready = pipeline.model is not None and pipeline.model.ocr is not None
    
    return {
        "status": "healthy" if model_ready else "degraded",
        "model": pipeline.config['model']['name'],
        "version": "1.0.0",
        "details": "Ready for real extraction" if model_ready else getattr(pipeline.model, 'load_error', 'Not ready')
    }

@app.post("/extract")
async def extract_kvps(file: UploadFile = File(...)):
    """Extract information from uploaded document"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    try:
        content = await file.read()
        result, hitl_required = await pipeline.extract(content, file.filename)
        
        return {
            "status": "review" if hitl_required else result.status,
            "kvps": result.kvps,
            "confidences": result.confidences,
            "hitl_required": hitl_required
        }
    except Exception as e:
        logger.exception("API extraction error")
        raise HTTPException(status_code=500, detail=str(e))
