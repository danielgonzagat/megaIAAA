from __future__ import annotations
import os, time
from typing import List, Dict, Any
import httpx
import json

class OpenAIClient:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        self.model = model or os.getenv("OPENAI_MODEL","gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cost_log = []

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        if not self.api_key:
            # fallback stub to keep framework plug-and-play even without keys
            return self._stub_reply(messages)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        t0 = time.time()
        with httpx.Client(timeout=60) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        dt = time.time()-t0
        text = data["choices"][0]["message"]["content"]
        # naive cost logging via usage
        usage = data.get("usage", {})
        self.cost_log.append({"ts": time.time(), "dt": dt, **usage})
        return text

    def _stub_reply(self, messages: List[Dict[str,str]]) -> str:
        # deterministic simple stub for offline use
        last = messages[-1]["content"] if messages else ""
        return f"[STUB/{self.model}] {last[:200]} :: answer synthesized."
