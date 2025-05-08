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
import json
import numpy as np
from ddt import ddt, unpack, idata

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.datasets import entanglement_concentration_data


@ddt
class TestEntangledConcentration(QiskitMachineLearningTestCase):
    """Entanglement Concentration Generator"""
    @idata([
        (3, "easy"),
        (3, "hard"),
        (4, "easy"),
        (4, "hard"),
        (8, "easy"),
        (8, "hard"),
    ])
    @unpack
    @unpack
    def test_default_params(self, n, mode):
        x_train, y_train, x_test, y_test = entanglement_concentration_data(
            training_size=4,
            test_size=4,
            n=n,
            mode=mode,
            one_hot=False,
        )
        np.testing.assert_array_equal(x_train.shape, (8, 2**n, 1))
        np.testing.assert_array_equal(x_test.shape, (8, 2**n, 1))
        np.testing.assert_array_almost_equal(y_train, np.array([0] * 4 + [1] * 4))
        np.testing.assert_array_almost_equal(y_test, np.array([0] * 4 + [1] * 4))

        # Now one_hot=True
        _, y_train_oh, _, y_test_oh = entanglement_concentration_data(
            training_size=4,
            test_size=4,
            n=n,
            mode=mode,
            one_hot=True,
        )
        np.testing.assert_array_equal(y_train_oh, np.array([[1, 0]] * 4 + [[0, 1]] * 4))
        np.testing.assert_array_equal(y_test_oh, np.array([[1, 0]] * 4 + [[0, 1]] * 4))




if __name__ == "__main__":
    unittest.main()
