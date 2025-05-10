# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
H Molecule Evolution
"""

from __future__ import annotations

import warnings
import os

import numpy as np
import pickle as pkl

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.exceptions import QiskitBackendNotFoundError

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


from ..utils import algorithm_globals


# pylint: disable=too-many-positional-arguments
def h_molecule_evolution_data(
    delta_t: float,
    train_end: int,
    test_start: int,
    test_end: int,
    molecule: str = "H2",
    noise_mode: str = "ibm_oslo",
    formatting: str = "ndarray"
) -> (
    tuple[Statevector, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[Statevector, np.ndarray, list[Statevector], np.ndarray, list[Statevector]]
):
    r""" """

    # Import Hamiltonian and Unitary Evolution Circuit
    occupancy = {"H2": 2, "H3": 2, "H6": 6}
    num_occupancy = occupancy[molecule]

    qc, hamiltonian = _evolution_circuit(molecule)
    qc_evo = qc.bind_parameters({t: delta_t})
    
    # Get Hartree Fock State
    psi_hf = _hartree_fock(hamiltonian, num_occupancy)

    # Noise Models for Training Data
    simulator = _noise_simulator(noise_mode)

    # Time stamps for Train & Test
    idx_train, idx_test = np.arange(0, train_end+1), np.arange(test_start, test_end+1)
    x_train, x_test = delta_t * idx_train, delta_t * idx_test


    return (psi_hf, x_train, _, x_test, _)


def _evolution_circuit(molecule):
    """Get the parametrized circuit for evolution after Trotterization. 
    Returns:
    - QuantumCircuit (for training set)
    - Parameter Object "t" (for training set)
    - Original Hamiltonian (for testing set)"""

    spo = _hamiltonian_import(molecule)
    
    t = Parameter("t")
    trotterizer = SuzukiTrotter(order=2, reps=1)
    u_evolution = PauliEvolutionGate(spo, time=t, synthesis=trotterizer)

    n_qubits = spo.num_qubits
    qc = QuantumCircuit(n_qubits)
    qc.append(u_evolution, range(n_qubits))

    qc_flat = qc.decompose()
    basis = ['rx', 'ry', 'rz', 'cx']

    qc_resolved = transpile(
        qc_flat,
        basis_gates=basis,
        optimization_level=3,
    )
    
    return qc_resolved, spo

def _hamiltonian_import(molecule):
    """Import Hamiltonian from Hamiltonians folder"""

    dir_path = os.path.dirname(__file__)
    filename = os.path.join(dir_path, f"hamiltonians\\{molecule}.bin")

    with open(filename, "rb") as f:
        spo = pkl.load(f)

    return spo

def _hartree_fock(hamiltonian, num_occupancy):
    """Finds an approximation of the Ground State for the Hamiltonian

    For Qubits being one-one mapped to Spin Orbitals, HF state is when
    all the lowest level orbitals are occupied with | 1 > state

    JW map automatically keeps orbitals in ascending order of energy"""
    
    n_qubits = hamiltonian.num_qubits

    bitstring = ['1']*num_occupancy ['0'] * (n_qubits - num_occupancy)

    occupation_label = ''.join(bitstring)

    return Statevector.from_label(occupation_label)

def _noise_simulator(noise_mode):
    """Returns a Noisy/Noiseless AerSimulator object"""

    if noise_level == "noiseless":
        noise_model = None
    
    elif noise_level == "reduced":
        single_qubit_error = depolarizing_error(0.001, 1)
        two_qubit_error = depolarizing_error(0.01, 2)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    # If the given Model is an IBM location
    else:    
        service = QiskitRuntimeService()
        
        try:
            backend = service.backend(noise_mode)
        except QiskitBackendNotFoundError:
            raise QiskitBackendNotFoundError(f"The specified backend '{noise_mode}' was not found.")
        
        noise_model = NoiseModel.from_backend(backend)
    
    simulator = AerSimulator(noise_model=noise_model)
    return simulator

print(h_molecule_evolution_data(1.0,3,6,8))