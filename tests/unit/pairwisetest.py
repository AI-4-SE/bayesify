import unittest
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline
from bayesify.pairwise import PyroMCMCRegressor, P4Preprocessing


class PreProcessingTests(unittest.TestCase):
    def test_constructor(self):
        P4Preprocessing()

    def test_preprocessing_fit(self):
        X, feature_names, y = get_X_y()
        pre = P4Preprocessing()
        pre.fit(X, y)
        return pre

    def test_fitting_preprocessing_transform(self):
        X, feature_names, y = get_X_y()
        pre = P4Preprocessing()
        new_X = pre.fit_transform(X, y)
        assertion_str = (
            "For the tips dataset, P4 "
            "preprocessing should create new "
            "features for interactions!"
        )
        self.assertGreater(
            len(new_X[0]),
            len(X[0]),
            assertion_str,
        )
        return new_X


class PWLearnerTests(unittest.TestCase):
    def test_constructor(self):
        reg = PyroMCMCRegressor()

    def test_repr(self):
        reg = PyroMCMCRegressor()
        print(reg)

    def test_fitting(self):
        reg = self.train_quick_model()

    def test_coef(self):
        reg = self.train_quick_model()
        coefs = reg.coef_
        self.assertIsNotNone(coefs)

    def test_prediction_samples(self):
        X, feature_names, y = get_X_y()
        reg = self.train_quick_model()
        n = 104
        predictions = reg.predict(X, n_samples=n)
        self.assertIsNotNone(predictions)
        self.assertEqual(
            n,
            len(predictions),
            "Did not get requested number of posterior predictive samples!",
        )

    def train_quick_model(self):
        X, feature_names, y = get_X_y()
        reg = PyroMCMCRegressor()
        mcmc_cores = 1
        reg.fit(
            X,
            y,
            mcmc_samples=100,
            mcmc_tune=200,
            feature_names=feature_names,
            mcmc_cores=mcmc_cores,
        )
        return reg

    def test_coefs_ci(self):
        reg = self.train_quick_model()
        coefs_50 = reg.coef_ci(0.5)
        coefs_95 = reg.coef_ci(0.95)

        root_width_50 = abs(coefs_50["root"][0] - coefs_50["root"][1])
        root_width_95 = abs(coefs_95["root"][0] - coefs_95["root"][1])
        self.assertGreater(root_width_95, root_width_50)

        for coef_50, coef_95 in zip(
            coefs_50["influences"].values(), coefs_95["influences"].values()
        ):
            width_50 = abs(coef_50[0] - coef_50[1])
            width_95 = abs(coef_95[0] - coef_95[1])
            self.assertGreater(width_95, width_50)


class PipelineTests(unittest.TestCase):
    def test_pipiline_fit(self):
        X, _, y = get_X_y()
        preproc = P4Preprocessing()
        reg = PyroMCMCRegressor(
            mcmc_samples=100,
            mcmc_tune=200,
        )
        pipeline = make_pipeline(preproc, reg)
        pipeline.fit(X, y)

    def train_quick_pipeline(self):
        X, feature_names, y = get_X_y()
        reg = PyroMCMCRegressor()
        mcmc_cores = 1
        reg.fit(
            X,
            y,
            mcmc_samples=100,
            mcmc_tune=200,
            feature_names=feature_names,
            mcmc_cores=mcmc_cores,
        )
        return reg


def get_X_y():
    tips = sns.load_dataset("tips")
    tips = pd.get_dummies(tips)
    y = tips["tip"].to_numpy()
    feature_names = ["total_bill", "sex_Male", "smoker_Yes", "size"]
    X = tips[feature_names].to_numpy()
    return X, feature_names, y


if __name__ == "__main__":
    unittest.main()
