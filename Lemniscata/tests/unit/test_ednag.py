import numpy as np
from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


def test_ednag_initialization_population_and_rates():
    config = IAAAConfig(ednag_population_size=4, ednag_mutation_rate=0.2, ednag_crossover_rate=0.7)
    system = IntegratedIAAASystem(config)
    ednag = system.components.get("ednag")
    assert ednag["type"] == "ednag"
    assert len(ednag["population"]) == 4
    assert ednag["mutation_rate"] == 0.2
    assert ednag["crossover_rate"] == 0.7
    # ensure each individual has required keys
    for individual in ednag["population"]:
        assert set(individual.keys()) >= {"id", "genome", "phenotype"}
