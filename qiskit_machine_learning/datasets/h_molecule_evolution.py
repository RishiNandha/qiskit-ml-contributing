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

from ..utils import algorithm_globals


# pylint: disable=too-many-positional-arguments
def h_molecule_evolution_data(
    molecule: str = "H2",
    test_size: int = 10,
    mode: str = "easy",
    one_hot: bool = True,
    include_sample_total: bool = False,
    sampling_method: str = "cardinal",
    class_labels: list | None = None,
    formatting: str = "ndarray",
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[list[Statevector], np.ndarray, list[Statevector], np.ndarray]
    | tuple[list[Statevector], np.ndarray, list[Statevector], np.ndarray, np.ndarray]
):
    r""" """

    # if include_sample_total:
    #     samples = np.array([n_points * 2])
    #     return (x_train, y_train, x_test, y_test, samples)

    return 0 #(x_train, y_train, x_test, y_test)


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
        optimization_level=1,
    )

    return qc_resolved, t, spo

def _hamiltonian_import(molecule):
    """Import Hamiltonian from Hamiltonians folder"""

    dir_path = os.path.dirname(__file__)
    filename = os.path.join(dir_path, f"hamiltonians\\{molecule}.bin")

    with open(filename, "rb") as f:
        spo = pkl.load(f)

    return spo

_evolution_circuit("H2")