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

""" Test Ad Hoc Data """

from test import QiskitMachineLearningTestCase

import unittest
import numpy as np
from ddt import ddt, unpack, idata

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import h_molecule_evolution


@ddt
class TestHMoleculeEvolution(QiskitMachineLearningTestCase):
    """H Molecule Evolution Tests"""
    
    @idata([Hx for Hx in ["H2", "H3", "H6"]])
    @unpack
    def test_default_params(self, molecule):
        """Checking for right shapes and labels"""
        HF, x_train, y_train, x_test, y_test = entanglement_concentration_data(
            delta_t = 1.0,
            train_end = 2,
            test_start = 4,
            test_end = 6,
            molecule = molecule
        )
        
        np.testing.assert_array_equal(x_train.shape, (8, 2**n, 1))
        np.testing.assert_array_equal(x_train.shape, (8, 2**n, 1))
        np.testing.assert_array_equal(x_test.shape, (8, 2**n, 1))
        np.testing.assert_array_almost_equal(y_train, np.array([0] * 4 + [1] * 4))
        np.testing.assert_array_almost_equal(y_test, np.array([0] * 4 + [1] * 4))


    @idata([(n,) for n in [3, 4]])
    @unpack
    def test_statevector_format(self, n):
        """Check if output values are normalized qiskit.circuit_info.Statevector objects"""
        x_train, _, _, _ = entanglement_concentration_data(
            training_size=4, test_size=1, n=n, formatting="statevector"
        )
        for state in x_train:
            self.assertIsInstance(state, Statevector)

            norm = np.linalg.norm(state.data)
            self.assertAlmostEqual(norm, 1.0, places=4)


    def test_error_raises(self):
        """Check if parameter errors are handled"""
        with self.assertRaises(ValueError):
            entanglement_concentration_data(training_size=4, test_size=1, n=1)

        with self.assertRaises(ValueError):
            entanglement_concentration_data(training_size=4, test_size=1, n=6)

if __name__ == "__main__":
    unittest.main()
