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

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit import ParameterVector

from ..utils import algorithm_globals


# pylint: disable=too-many-positional-arguments
def h_molecule_evolution_data(
    training_size: int,
    test_size: int,
    n: int,
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
    r"""
    """

    if include_sample_total:
        samples = np.array([n_points * 2])
        return (x_train, y_train, x_test, y_test, samples)

    return (x_train, y_train, x_test, y_test)

