"""Job orchestrator consuming adapters."""
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, Dict, List

from src.lemnisiana.adapters import Adapter


class Mode(str, Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    EVOLVE = "evolve"


@dataclass
class Job:
    adapter: Adapter
    context: Dict[str, Any]
    mode: Mode


class Orchestrator:
    def __init__(self, promotion_threshold: float = 0.5) -> None:
        self._queue: "Queue[Job]" = Queue()
        self.promotion_threshold = promotion_threshold
        self.promoted: List[Dict[str, Any]] = []

    def submit(self, job: Job) -> None:
        """Submit a job to the queue."""
        self._queue.put(job)

    def run(self) -> List[Dict[str, Any]]:
        """Process all pending jobs and return their results."""
        results: List[Dict[str, Any]] = []
        while not self._queue.empty():
            job = self._queue.get()
            result = self._run_job(job)
            if result.get("score", 0) >= self.promotion_threshold:
                self.promoted.append(result)
            results.append(result)
        return results

    def _run_job(self, job: Job) -> Dict[str, Any]:
        adapter = job.adapter
        context = job.context
        if job.mode is Mode.EXPLORE:
            proposal = adapter.propose(context)
            score = adapter.grade(proposal)
            return {"proposal": proposal, "score": score}
        if job.mode is Mode.EXPLOIT:
            trained = adapter.train(context)
            optim = adapter.optimize(trained)
            score = adapter.grade(optim)
            return {"optimized": optim, "score": score}
        # EVOLVE
        proposal = adapter.propose(context)
        trained = adapter.train(proposal)
        optim = adapter.optimize(trained)
        score = adapter.grade(optim)
        return {"optimized": optim, "score": score}
