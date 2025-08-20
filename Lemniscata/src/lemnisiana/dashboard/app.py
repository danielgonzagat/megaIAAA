from __future__ import annotations
import os, json, time
from fastapi import FastAPI
from pydantic import BaseModel
from ..metrics.calibration import ece_from_log

app = FastAPI(title="Lemnisiana Dashboard")

CAL_LOG = []

class CalEntry(BaseModel):
    prob: float
    acc: float

@app.get("/health")
def health(): return {"ok": True, "ts": time.time()}

@app.post("/log/calibration")
def log_cal(e: CalEntry):
    CAL_LOG.append(e.model_dump())
    return {"n": len(CAL_LOG), "ece": ece_from_log(CAL_LOG)}

@app.get("/metrics/ece")
def metrics_ece(bins: int = 10):
    return {"bins": bins, "ece": ece_from_log(CAL_LOG, bins=bins)}

def cli():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
