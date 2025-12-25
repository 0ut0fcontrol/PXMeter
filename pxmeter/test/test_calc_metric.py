# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import unittest

from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.eval import evaluate
from pxmeter.test.test_utils import TEST_DATA_DIR


class TestCalcMetric(unittest.TestCase):
    """
    Test class for calculating metrics and comparing results.
    """

    def setUp(self) -> None:
        self._start_time = time.time()
        super().setUp()

    def tearDown(self):
        elapsed_time = time.time() - self._start_time
        logging.info("Test %s took %.6f seconds", self.id(), elapsed_time)

    def test_metric(self):
        """
        Test the evaluation function with DockQ and LDDT metrics.
        """

        ref_cif = TEST_DATA_DIR / "7n0a_ref.cif"
        model_cif = TEST_DATA_DIR / "7n0a_model.cif"

        result = evaluate(
            ref_cif=ref_cif,
            model_cif=model_cif,
            run_config=RUN_CONFIG,
        )
        result_dict = result.to_json_dict()

        # Check DockQ result (Ground Truth from official DockQ implementation)
        # Interface mapping based on chain map: {"A": "B0", "B": "C0", "C": "A0"}
        # Ref A-B -> Model B0-C0 (interface "B,C")
        # Ref A-C -> Model B0-A0 (interface "A,B")
        # Ref B-C -> Model C0-A0 (interface "A,C")

        self.assertAlmostEqual(
            0.84754176,
            result_dict["interface"]["B,C"]["dockq"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            0.71798609,
            result_dict["interface"]["A,B"]["dockq"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            0.66608325,
            result_dict["interface"]["A,C"]["dockq"],
            delta=1e-6,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
