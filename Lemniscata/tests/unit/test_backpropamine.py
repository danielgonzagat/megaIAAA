from lemnisiana.legacy_core.iaaa_integrated_system import IAAAConfig, IntegratedIAAASystem


def test_backpropamine_initialization_and_history():
    config = IAAAConfig(initial_learning_rate=0.005, adaptation_strength=0.2)
    system = IntegratedIAAASystem(config)
    bp = system.components.get("backpropamine")
    assert bp["type"] == "backpropamine"
    assert bp["learning_rate"] == 0.005
    assert bp["adaptation_strength"] == 0.2
    # simulate gradients to update history
    bp["gradient_history"].extend([0.1, -0.2, 0.05])
    system._update_backpropamine(bp)
    assert bp["learning_rate_history"]
