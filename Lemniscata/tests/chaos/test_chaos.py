import pytest
from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


@pytest.mark.skip(reason="Chaos test scaffold - enable to simulate fault tolerance scenarios")
def test_chaos_component_failure():
    config = IAAAConfig()
    system = IntegratedIAAASystem(config)
    # Simulate component failure by removing EDNAG
    system.components.pop("ednag")
    system.evolve(num_cycles=1)
