import unittest
import pandas as pd
import sys

sys.path.append("/home/graeme/process_uq/uq_tools")
from confidence_analysis import ConfidenceAnalysis
from uncertainty_data import UncertaintyData


class TestConfidenceAnalysis(unittest.TestCase):

    def setUp(self):
        # Set up any data or configurations needed for tests
        self.uq_data = UncertaintyData(
            path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
            figure_of_merit="sqsumsq",
            input_parameters=["aspect", "tbrnmn"],
        )
        self.input_names = ["aspect", "tbrnmn"]
        self.number_intervals = 2
        self.weight_confidence = 1.0
        self.weight_overlap = 0.5

    def test_initialization(self):
        # Test initialization of ConfidenceAnalysis instance

        confidence_analysis = ConfidenceAnalysis(
            uq_data=self.uq_data,
            input_names=self.input_names,
            number_intervals=self.number_intervals,
            weight_confidence=self.weight_confidence,
            weight_overlap=self.weight_overlap,
        )
        self.assertIsInstance(confidence_analysis, ConfidenceAnalysis)
        self.assertEqual(confidence_analysis.number_intervals, self.number_intervals)
        self.assertEqual(confidence_analysis.weight_confidence, self.weight_confidence)
        self.assertEqual(confidence_analysis.weight_overlap, self.weight_overlap)
        self.assertEqual(confidence_analysis.uq_data, self.uq_data)
        self.assertEqual(confidence_analysis.input_names, self.input_names)

    # def test_calculate_metric(self):
    #     # Test the calculate_metric method
    #     confidence_analysis = ConfidenceAnalysis(
    #         uq_data=self.uq_data,
    #         input_names=self.input_names,
    #         number_intervals=self.number_intervals,
    #         weight_confidence=self.weight_confidence,
    #         weight_overlap=self.weight_overlap,
    #     )
    #     confidences = [0.8, 0.7, 0.9]
    #     errors = [0.1, 0.05, 0.15]
    #     expected_metric = 1.0 * sum(confidences) - 0.5 * confidence_analysis.overlap(
    #         confidences, errors
    #     )
    #     calculated_metric = confidence_analysis.calculate_metric(
    #         confidences, errors, self.weight_confidence, self.weight_overlap
    #     )
    #     self.assertAlmostEqual(calculated_metric, expected_metric, places=5)


if __name__ == "__main__":
    unittest.main()
