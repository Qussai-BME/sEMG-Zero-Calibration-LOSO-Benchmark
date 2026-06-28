#!/usr/bin/env python3
"""
api.py - FastAPI REST API for EMG Analysis Engine.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import io
import json
from typing import Optional

from core_engine import EMGFeatureExtractor, EMGConfig

app = FastAPI(title="EMG Analysis Engine API", version="1.0.0")


@app.post("/analyze", response_class=JSONResponse)
async def analyze(
    file: UploadFile = File(...),
    sampling_rate: int = Form(2000),
    cutoff_low: float = Form(20.0),
    cutoff_high: float = Form(450.0),
    filter_order: int = Form(4),
    notch_freq: float = Form(50.0),
    window_ms: int = Form(100),
    overlap: float = Form(0.5),
    filter_type: str = Form("butterworth"),
    noise_method: str = Form("percentile"),
    psd_method: str = Form("welch"),
    compute_freq: bool = Form(False)
):
    """
    Upload an EMG file (CSV, TXT, NPY) and receive analysis results as JSON.
    """
    contents = await file.read()

    # Determine file type and load data
    if file.filename.endswith('.csv'):
        data = np.loadtxt(io.BytesIO(contents), delimiter=',')
    elif file.filename.endswith('.txt'):
        data = np.loadtxt(io.BytesIO(contents))
    elif file.filename.endswith('.npy'):
        data = np.load(io.BytesIO(contents))
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    config = EMGConfig(
        sampling_rate=sampling_rate,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        filter_order=filter_order,
        notch_freq=notch_freq,
        window_size=int(window_ms * sampling_rate / 1000),
        overlap=overlap,
        filter_type=filter_type,
        noise_estimation_method=noise_method,
        psd_method=psd_method
    )

    engine = EMGFeatureExtractor(config)
    try:
        results = engine.process_stream(data, compute_freq_features=compute_freq)
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}