from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


def test_interaction_ednag_to_backpropamine_exists():
    config = IAAAConfig()
    system = IntegratedIAAASystem(config)
    interactions = system.component_interactions["ednag"]
    assert any(interaction["target"] == "backpropamine" for interaction in interactions)
