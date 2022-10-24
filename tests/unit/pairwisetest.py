import unittest
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from bayesify.pairwise import PWRegressor


class SKLearnInterfaceTest(unittest.TestCase):
    def test_something(self):
        tips = sns.load_dataset('tips')
        tips = pd.get_dummies(tips)
        y = tips["tip"]
        feature_names = ["total_bill", "sex_Male", "smoker_Yes", "size"]
        X = tips[feature_names]
        reg = PWRegressor()
        mcmc_cores = 1
        reg.fit(X, y, mcmc_samples=100, mcmc_tune=200, feature_names=feature_names, mcmc_cores=mcmc_cores)

        # self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
