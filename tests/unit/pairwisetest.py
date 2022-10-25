import unittest
import seaborn as sns
import pandas as pd

from bayesify.pairwise import PWRegressor


class SKLearnInterfaceTest(unittest.TestCase):

    def test_constructor(self):
        reg = PWRegressor()
    def test_repr(self):
        reg = PWRegressor()
        print(reg)

    def test_fitting(self):
        X, feature_names, y = self.get_X_y()
        reg = PWRegressor()
        mcmc_cores = 1
        reg.fit(X, y, mcmc_samples=100, mcmc_tune=200, feature_names=feature_names, mcmc_cores=mcmc_cores)

        # self.assertEqual(True, False)  # add assertion here

    def test_coef(self):
        X, feature_names, y = self.get_X_y()
        reg = PWRegressor()
        mcmc_cores = 1
        reg.fit(X, y, mcmc_samples=100, mcmc_tune=200, feature_names=feature_names, mcmc_cores=mcmc_cores)
        coefs = reg.coef_
        self.assertIsNotNone(coefs)

    def test_coefs_ci(self):
        X, feature_names, y = self.get_X_y()
        reg = PWRegressor()
        mcmc_cores = 1
        reg.fit(X, y, mcmc_samples=100, mcmc_tune=200, feature_names=feature_names, mcmc_cores=mcmc_cores)
        coefs_50 = reg.coef_ci(0.5)
        coefs_95 = reg.coef_ci(0.95)

        root_width_50 = abs(coefs_50["root"][0] - coefs_50["root"][1])
        root_width_95 = abs(coefs_95["root"][0] - coefs_95["root"][1])
        self.assertGreater(root_width_95, root_width_50)

        for coef_50, coef_95 in zip(coefs_50["influences"].values(), coefs_95["influences"].values()):
            width_50 = abs(coef_50[0] - coef_50[1])
            width_95 = abs(coef_95[0] - coef_95[1])
            self.assertGreater(width_95, width_50)

    def get_X_y(self):
        tips = sns.load_dataset('tips')
        tips = pd.get_dummies(tips)
        y = tips["tip"]
        feature_names = ["total_bill", "sex_Male", "smoker_Yes", "size"]
        X = tips[feature_names]
        return X, feature_names, y


if __name__ == '__main__':
    unittest.main()
