import unittest
import pandas as pd
import sys

sys.path.append("/home/graeme/process_uq/uq_tools")
from uncertainty_optimisation import UncertaintyOptimisation
from uncertainty_data import UncertaintyData


class TestUncertaintyOptimisation(unittest.TestCase):

    def setUp(self):
        # Set up any data or configurations needed for tests
        self.uq_data = UncertaintyData(
            path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
            sampled_variables=["aspect", "tbrnmn"],
        )
        self.input_names = ["aspect", "tbrnmn"]
        self.number_intervals = 2
        self.weight_confidence = 1.0
        self.weight_overlap = 0.5

    def test_initialization(self):
        # Test initialization of UncertaintyOptimisation instance

        uncertainty_optimisation = UncertaintyOptimisation(
            uq_data=self.uq_data,
            input_names=self.input_names,
            number_intervals=self.number_intervals,
            weight_confidence=self.weight_confidence,
            weight_overlap=self.weight_overlap,
        )
        self.assertIsInstance(uncertainty_optimisation, UncertaintyOptimisation)
        self.assertEqual(
            uncertainty_optimisation.number_intervals, self.number_intervals
        )
        self.assertEqual(
            uncertainty_optimisation.weight_confidence, self.weight_confidence
        )
        self.assertEqual(uncertainty_optimisation.weight_overlap, self.weight_overlap)
        self.assertEqual(uncertainty_optimisation.uq_data, self.uq_data)
        self.assertEqual(uncertainty_optimisation.input_names, self.input_names)

    # def test_calculate_metric(self):
    #     # Test the calculate_metric method
    #     uncertainty_optimisation = UncertaintyOptimisation(
    #         uq_data=self.uq_data,
    #         input_names=self.input_names,
    #         number_intervals=self.number_intervals,
    #         weight_confidence=self.weight_confidence,
    #         weight_overlap=self.weight_overlap,
    #     )
    #     confidences = [0.8, 0.7, 0.9]
    #     errors = [0.1, 0.05, 0.15]
    #     expected_metric = 1.0 * sum(confidences) - 0.5 * uncertainty_optimisation.overlap(
    #         confidences, errors
    #     )
    #     calculated_metric = uncertainty_optimisation.calculate_metric(
    #         confidences, errors, self.weight_confidence, self.weight_overlap
    #     )
    #     self.assertAlmostEqual(calculated_metric, expected_metric, places=5)


if __name__ == "__main__":
    unittest.main()
