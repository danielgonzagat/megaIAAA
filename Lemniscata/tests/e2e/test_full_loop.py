from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


def test_full_lemniscata_loop_one_cycle():
    config = IAAAConfig(ednag_population_size=3, max_evolution_cycles=1)
    system = IntegratedIAAASystem(config)
    results = system.evolve(num_cycles=1)
    assert results["cycles_completed"] == 1
    assert "final_performance" in results
    assert isinstance(results["evolution_trajectory"], list)
