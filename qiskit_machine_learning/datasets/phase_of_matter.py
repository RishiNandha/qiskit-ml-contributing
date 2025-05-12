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
Phase Of Matter
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
def phase_of_matter_data(
    training_size: int, test_size: int, n: int
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[list[Statevector], np.ndarray, list[Statevector], np.ndarray]
    | tuple[list[Statevector], np.ndarray, list[Statevector], np.ndarray, np.ndarray]
):
    r"""
    Parameters:
        training_size : Number of training samples per class.
        test_size :  Number of testing samples per class.
        n : Number of qubits (dimension of the feature space). Current implementation
            supports only 3, 4 and 8
        mode :
            Choices are:

                * ``"easy"``: uses CE values 0.18 and 0.40 for n = 3 and 0.12 and 0.43 for n = 4
                * ``"hard"``: uses CE values 0.28 and 0.40 for n = 3 and 0.22 and 0.34 for n = 4

            Default is ``"easy"``.
        one_hot : If True, returns labels in one-hot format. Default is True.
        include_sample_total : If True, the function also returns the total number
            of accepted samples. Default is False.
        sampling_method: The method used to generate input states.
            Choices are:

                * ``"isotropic"``: samples qubit states uniformly in the bloch sphere
                * ``"cardinal"``: samples qubit states out of the 6 axes of bloch sphere

            Default is ``"cardinal"``.
        class_labels : Custom labels for the two classes when one-hot is not enabled.
            If not provided, the labels default to ``0`` and ``+1``
        formatting: The format in which datapoints are given.
            Choices are:

                * ``"ndarray"``: gives a numpy array of shape (n_points, 2**n_qubits, 1)
                * ``"statevector"``: gives a python list of Statevector objects

            Default is ``"ndarray"``.

    Returns:
        Tuple
        containing the following:

        * **training_features** : ``np.ndarray`` | ``qiskit.quantum_info.Statevector``
        * **training_labels** : ``np.ndarray``
        * **testing_features** : ``np.ndarray`` | ``qiskit.quantum_info.Statevector``
        * **testing_labels** : ``np.ndarray``

        If ``include_sample_total=True``, a fifth element (``np.ndarray``) is included
        that specifies the total number of accepted samples.
    """
    return
