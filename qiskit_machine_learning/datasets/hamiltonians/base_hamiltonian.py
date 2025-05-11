"""
Base class for quantum spin chain Hamiltonian models.

This module provides the foundation for implementing various spin chain models,
with methods for creating and sampling Hamiltonians with different parameters.
"""

from qiskit.quantum_info import SparsePauliOp
import numpy as np

class HamiltonianModel:
    """Base class for quantum spin chain Hamiltonian models.

    This class provides the foundation for implementing various spin chain models,
    with methods for creating and sampling Hamiltonians with different parameters.

    Attributes:
        num_qubits (int): Number of qubits (spins) in the system.
    """

    def __init__(self, num_qubits):
        """Initialize the Hamiltonian model.

        Args:
            num_qubits (int): Number of qubits in the system.
        """
        self.num_qubits = num_qubits

    def get_hamiltonian(self):
        """Get the Hamiltonian of the model as a SparsePauliOp.

        Returns:
            SparsePauliOp: The Hamiltonian operator.
        """
        raise NotImplementedError("Subclasses must implement get_hamiltonian method")

    def sample_parameters(self, **kwargs):
        """Sample model parameters from specified ranges.

        Returns:
            list: List of sampled model instances.
        """
        raise NotImplementedError("Subclasses must implement sample_parameters method")
        
    def get_phase(self):
        """Determine the phase of the model based on parameters.

        Returns:
            str: The phase label.
        """
        raise NotImplementedError("Subclasses must implement get_phase method")
