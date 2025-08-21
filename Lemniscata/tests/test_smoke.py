def test_import():
    import lemnisiana

    from lemnisiana.ednag import ArchitectureGenerator
    from lemnisiana.backpropamine import NeuromodulatedNetwork
    from lemnisiana.rdis import RDISPipeline
    from lemnisiana.autopoietic import AutopoieticNetwork
    from lemnisiana.quantum import QuantumCircuit

    assert ArchitectureGenerator
    assert NeuromodulatedNetwork
    assert RDISPipeline
    assert AutopoieticNetwork
    assert QuantumCircuit
