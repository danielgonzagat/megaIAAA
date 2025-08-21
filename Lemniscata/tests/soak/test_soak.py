import pytest
from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


@pytest.mark.skip(reason="Soak test scaffold - enable for long-running stability tests")
def test_soak_runs_multiple_cycles():
    config = IAAAConfig(max_evolution_cycles=100, ednag_population_size=10)
    system = IntegratedIAAASystem(config)
    system.evolve(num_cycles=100)
