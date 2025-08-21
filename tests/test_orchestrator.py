from orchestrator import Orchestrator, Job, Mode
from src.lemnisiana.adapters import EDNAGAdapter, BackpropamineAdapter


def test_e2e_loop_promotes_candidate():
    adapter = EDNAGAdapter()
    orch = Orchestrator(promotion_threshold=0.5)
    orch.submit(Job(adapter=adapter, context={"seed": 42}, mode=Mode.EVOLVE))
    results = orch.run()
    assert len(results) == 1
    result = results[0]
    # verify full pipeline executed
    assert result["optimized"]["trained"]
    assert result["optimized"]["optimized"]
    # candidate should be promoted due to high score
    assert orch.promoted == [result]


def test_multiple_modes():
    adapter = BackpropamineAdapter()
    orch = Orchestrator(promotion_threshold=0.9)
    orch.submit(Job(adapter=adapter, context={"seed": 1}, mode=Mode.EXPLORE))
    orch.submit(Job(adapter=adapter, context={"seed": 1}, mode=Mode.EXPLOIT))
    results = orch.run()
    # explore should only provide proposal
    assert "proposal" in results[0]
    # exploit should return optimized candidate
    assert "optimized" in results[1]
