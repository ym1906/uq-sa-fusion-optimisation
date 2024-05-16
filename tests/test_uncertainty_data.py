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
            sampled_variables=["aspect", "tbrnmn"],
        )
        merged_df = uq_data.merge_hdf_files()
        self.assertIsInstance(merged_df, pd.DataFrame)

    def test_calculate_sensitivity(self, sampled_variables):
        # Test if sensitivity calculation returns a DataFrame
        uq_data = UncertaintyData(
            path_to_uq_data_folder="/home/graeme/easyVVUQ-process/demo_runs_2/run1/",
            sampled_variables=["aspect", "tbrnmn"],
        )
        uq_data.uncertainties_df = self.test_data
        uq_data.calculate_sensitivity(figure_of_merit="rmajor")
        self.assertIsInstance(uq_data.sensitivity_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
