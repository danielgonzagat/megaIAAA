# Lemnisiana IAAA — R33 All‑in‑One

**Plug‑and‑play** package that wires R‑Zero on‑policy with multi‑metric self‑consistency, TTD‑DR with active retrieval, Dual‑Verifier (LLM grader + heuristics), adapters (LoRA/FT) with **shadow→canary→promotion** gates, full benchmarks (GSM8K/MMLU/MBPP loaders), calibration (ECE) and a Prometheus/Grafana‑ready dashboard.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # editable
export OPENAI_API_KEY=sk-...
# Optional: set LLM provider (stub|openai|ollama)
export LEM_PROVIDER=openai
# Launch dashboard (http://localhost:8000)
lemnisiana-dashboard
```

### Run R‑Zero on‑policy (self-consistency maj@k)
```bash
lemnisiana-rzero --maj_k 7 --samples 8 --verifier real --dataset ./datasets/gsm8k/train.jsonl
```

### Run TTD‑DR with active retrieval
```bash
lemnisiana-ttddr --corpus ./corpus --steps 12 --emit report.md
```

### Train LoRA adapter from fail‑cases (auto‑curriculum)
```bash
lemnisiana-train-lora --failures ./artifacts/failures.jsonl --base_model Qwen2.5-3B
```

Security tips: run canaries with CPU/mem quotas and a seccomp profile:
```bash
docker run --rm -p 8000:8000   --cpus="2" --memory="4g"   --security-opt seccomp=./docker/seccomp-profile.json   lemnisiana:latest
```
