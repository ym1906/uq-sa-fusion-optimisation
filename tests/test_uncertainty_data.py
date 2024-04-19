import unittest
import pandas as pd
import sys

sys.path.append("/home/graeme/process_uq/uq_tools")
from uncertainty_data import UncertaintyData


class TestUncertaintyData(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        self.test_data = pd.DataFrame(
            {
                "param1": "aspect",
                "param2": "tbrnmn",
                "sqsumsq": [0.1, 0.2, 0.3],
                "ifail": [1, 0, 1],  # For testing converged and unconverged data
            }
        )

    def test_merge_hdf_files(self):
        # Test if merging HDF files returns a DataFrame
        uq_data = UncertaintyData(
            path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
            figure_of_merit="sqsumsq",
            input_parameters=["aspect", "tbrnmn"],
        )
        merged_df = uq_data.merge_hdf_files()
        self.assertIsInstance(merged_df, pd.DataFrame)

    def test_calculate_sensitivity(self):
        # Test if sensitivity calculation returns a DataFrame
        uq_data = UncertaintyData(
            path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
            figure_of_merit="sqsumsq",
            input_parameters=["aspect", "tbrnmn"],
        )
        uq_data.uncertainties_df = self.test_data
        uq_data.calculate_sensitivity(figure_of_merit="rmajor")
        self.assertIsInstance(uq_data.sensitivity_df, pd.DataFrame)

    # def test_find_significant_parameters(self):
    #     # Test if significant parameters are found
    #     uq_data = UncertaintyData(
    #         path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
    #         figure_of_merit="sqsumsq",
    #         input_parameters=["aspect", "tbrnmn"],
    #     )
    #     uq_data.sensitivity_df = pd.DataFrame(
    #         {"S1": [0.1, 0.05]}, index=["aspect", "tbrnmn"]
    #     )
    #     significant_params = uq_data.find_significant_parameters(
    #         uq_data.sensitivity_df, "S1", significance_level=0.05
    #     )
    #     self.assertEqual(significant_params, ["aspect"])

    # def test_estimate_design_values(self):
    #     # Test if design values are estimated correctly
    #     uq_data = UncertaintyData(
    #         path_to_uq_data_folder="path/to/folder",
    #         figure_of_merit="sqsumsq",
    #         input_parameters=["param1", "param2"],
    #     )
    #     design_values = uq_data.estimate_design_values(["param1", "param2"])
    #     self.assertIsInstance(design_values, pd.Series)


if __name__ == "__main__":
    unittest.main()
