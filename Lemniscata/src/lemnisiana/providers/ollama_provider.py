from __future__ import annotations
import os
from typing import List, Dict
import ollama

class OllamaClient:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL","llama3.1")

    def chat(self, messages: List[Dict[str,str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        try:
            resp = ollama.chat(model=self.model, messages=messages, options={"temperature": temperature, "num_predict": max_tokens})
            return resp["message"]["content"]
        except Exception as e:
            last = messages[-1]["content"] if messages else ""
            return f"[OLLAMA-STUB/{self.model}] {last[:200]} :: {e}"
