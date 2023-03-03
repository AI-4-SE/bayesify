# import networkx as nx
import copy
import datetime
import itertools
import string
import time
from math import sqrt
import os
import platform
import math
from string import ascii_lowercase

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoLars
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import bz2
import pickle
import sys

from xml.etree import ElementTree as ET

import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
import jax.numpy as jnp
import jax
from jax import random
from pprint import pprint, pformat
from sklearn.pipeline import make_pipeline
from bayesify.datahandler import DistBasedRepo
from itertools import product, islice


def all_words():
    alphabet = string.ascii_lowercase
    l = 1
    while True:
        for alph_tuple in product(alphabet, repeat=l):
            new_string = "".join(alph_tuple)
            yield new_string
        l += 1


def get_n_words(n_options):
    return list(islice(all_words(), n_options))


class P4Preprocessing(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        inters_only_between_influentials=True,
        prior_broaden_factor=1,
        t_wise=None,
        rnd_seed=0,
        verbose=False,
    ):
        self.pos_map = None
        self.cost_ft_selection = None
        self.final_var_names = None
        self.interactions_possible = None
        self.feature_names = None
        self.inters_only_between_influentials = inters_only_between_influentials
        self.prior_broaden_factor = prior_broaden_factor
        self.t_wise = t_wise
        self.rnd_seed = rnd_seed
        self.verbose = verbose
        self.feature_names_out = None

    def fit(self, X, y, model_interactions=True, feature_names=None, pos_map=None):
        n_options = len(X[0])
        if feature_names:
            self.feature_names = feature_names
            self.pos_map = {opt: idx for idx, opt in enumerate(self.feature_names)}
        elif pos_map:
            self.pos_map = pos_map
            self.feature_names = list(pos_map)
        else:
            self.feature_names = get_n_words(n_options)
            self.pos_map = {opt: idx for idx, opt in enumerate(self.feature_names)}
        if model_interactions:
            if self.t_wise:
                self.interactions_possible = self.t_wise > 1
                self.print("Interactions possible because t =", self.t_wise)
            else:
                self.interactions_possible = n_options < len(X)
        else:
            self.interactions_possible = False

        start_ft_selection = time.time()
        self.print("Starting feature and interaction selection.")
        self.final_var_names, _, _ = self.get_influentials_from_lasso(X, y)
        self.feature_names_out = list(self.final_var_names.keys())
        assert self.final_var_names, (
            "Lasso feature selection selected no options of interactions. "
            "Hence, we cannot learn any influence!"
        )
        self.cost_ft_selection = time.time() - start_ft_selection
        self.print(
            "Feature selection with lasso took {}s".format(self.cost_ft_selection)
        )

        return self

    def transform(self, X):
        rv_names, X = self.get_p4_train_data(X)
        return X

    def fit_transform(self, X, y=None, *fit_args, **fit_params):
        if y is None:
            print("Need y to do preprocessing! Returning untouched X.")
            return X
        fitted_preproc = self.fit(X, y, *fit_args, **fit_params)
        transformed_X = fitted_preproc.transform(X)
        return transformed_X
        #
        # if y is not None:
        #     return X, y
        # else:
        #     return X

    def transform_data_to_candidate_features(self, candidate, train_x):
        mapped_features = []
        for term in candidate:
            idx = [self.pos_map[ft] for ft in term]
            selected_cols = np.array(train_x)[:, idx]
            if len(idx) > 1:
                mapped_feature = np.product(selected_cols, axis=1).ravel()
            else:
                mapped_feature = selected_cols.ravel()
            mapped_features.append(list(mapped_feature))
        reshaped_mapped_x = np.atleast_2d(mapped_features).T
        return reshaped_mapped_x

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def generate_valid_combinations(self, first_stage_influential_ft, all_ft):
        print_flush("Generating Interaction Terms")
        if self.inters_only_between_influentials:
            all_inter_pairs = list(
                itertools.combinations(first_stage_influential_ft, 2)
            )
        else:
            all_inter_pairs = list(
                itertools.product(first_stage_influential_ft, all_ft)
            )
        valid_pairs = []
        print("Computing x values for", len(all_inter_pairs), "interactions")
        sys.stdout.flush()
        for a, b in all_inter_pairs:
            idx_a = self.pos_map[a]
            idx_b = self.pos_map[b]
            x_np = self.x_shared.get_value()
            vals_a_np = np.array(list(x_np[:, idx_a]))
            vals_b_np = np.array(list(x_np[:, idx_b]))
            is_non_constant = self.not_constant_term_cheap(vals_a_np, vals_b_np, x_np)
            if is_non_constant:
                valid_pairs.append((a, b))
        print_flush("Checked all interactions for constance ones.")
        return valid_pairs

    def not_constant_term(self, vals_a_np, vals_b_np):
        vals_prod = np.multiply(vals_a_np, vals_b_np)
        is_non_constant = len(np.unique(vals_prod)) > 1
        return is_non_constant

    def not_constant_term_cheap(self, vals_a_np, vals_b_np, train_set, slice_size=500):
        n_samples = len(vals_a_np)
        is_non_constant = False
        duplicates_any_features = True
        for slice_start in range(0, n_samples, slice_size):
            slice_end = min(slice_start + slice_size, n_samples)
            small_a = vals_a_np[slice_start:slice_end]
            small_b = vals_b_np[slice_start:slice_end]
            small_train_set = train_set[slice_start:slice_end]
            vals_prod = np.multiply(small_a, small_b)

            if duplicates_any_features:
                duplicates_any_features = np.any(
                    [
                        np.all(vals_prod == small_train_set[:, i])
                        for i in range(small_train_set.shape[1])
                    ]
                )

            if not is_non_constant:
                is_non_constant = len(np.unique(vals_prod)) > 1
            if is_non_constant and not duplicates_any_features:
                return True
        return False

    def vector_inner_prod_slow(self, a, b, slice_size=1000):
        length = len(a)
        products = []
        for slice_start in range(0, length, slice_size):
            slice_end = min(slice_start + slice_size, length)
            small_a = a[slice_start:slice_end]
            small_b = b[slice_start:slice_end]
            vals_prod = np.prod([small_a, small_b], axis=0)
            products.append(vals_prod)
        joint_prod = np.concatenate(products)
        return joint_prod

    def get_ft_and_inters_from_rvs(
        self, inter_trace, noise_str, p_mass, ft_inter_names
    ):
        significant_inter_ft = self.get_significant_fts(
            inter_trace, noise_str, p_mass, ft_inter_names
        )
        ft_or_inter = [
            a.replace("influence_", "").split("&")
            for a in significant_inter_ft
            if "root" not in a
        ]
        final_ft = [ft[0] for ft in ft_or_inter if len(ft) == 1]
        final_inter = [ft for ft in ft_or_inter if len(ft) > 1]
        return final_ft, final_inter

    def get_priors_from_lin_reg(self, rv_names, sd_scale=None):
        if sd_scale is None:
            sd_scale = self.prior_broaden_factor
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(rv_names)
        if not self.no_plots:
            self.save_spectrum_fig(reg_dict_final, err_dict, rv_names)

        all_raw_errs = [errs["raw"] for errs in list(err_dict.values())]
        all_abs_errs = np.array(
            [abs(err["y_pred"] - err["y_true"]) for err in all_raw_errs]
        )

        noise_sd_over_all_regs = sd_scale * 2 * float(all_abs_errs.mean())
        reg_list = list(reg_dict_final.values())
        alphas = []
        betas = []
        for coef_id, _ in enumerate(reg_list[0].coef_):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list])
            alpha, beta = norm.fit(
                coef_candidates,
            )
            alpha = max(coef_candidates)
            alphas.append(alpha)
            betas.append(beta)
        prior_coef_means = np.array(alphas)
        prior_coef_stdvs = np.array(betas)
        prior_root_mean, prior_root_std = norm.fit(
            np.array([reg.intercept_ for reg in reg_list]),
        )

        return (
            noise_sd_over_all_regs,
            prior_coef_means,
            prior_coef_stdvs,
            prior_root_mean,
            prior_root_std,
        )

    def get_priors_from_train_set(self, rv_names, sd_scale, n_influentials=10):
        mean_perf = np.mean(self.y)
        expected_ft_mean = mean_perf / n_influentials
        expected_std = expected_ft_mean / n_influentials * sd_scale
        mean_priors = np.array([expected_ft_mean] * len(rv_names))
        std_priors = np.array([expected_std] * len(rv_names))
        noise_sd = expected_ft_mean * sd_scale

        root_mean_prior = mean_perf / 2
        root_std_prior = mean_perf / 2 * sd_scale

        return noise_sd, mean_priors, std_priors, root_mean_prior, root_std_prior

    def get_uninformed_priors_from_train_set(self, y, rv_names):
        mean_perf = y
        expected_ft_mean = 0
        expected_std = mean_perf * 10
        mean_priors = np.array([expected_ft_mean] * len(rv_names))
        std_priors = np.array([expected_std] * len(rv_names))
        noise_sd = expected_std

        root_mean_prior = expected_ft_mean
        root_std_prior = expected_std

        return noise_sd, mean_priors, std_priors, root_mean_prior, root_std_prior

    def get_significant_fts(self, lin_trace, noise_str, p_mass, ft_inter_names):
        significant_ft = []
        for ft in lin_trace.varnames:
            if (
                "_log__" not in ft
                and "relative_error" not in ft
                and noise_str not in ft
                and "root" not in ft
                and "active_" not in ft
                and "_scale" not in ft
            ):
                if lin_trace[ft].shape[1] == 1:
                    mass_less_zero = (lin_trace[ft] > 0).mean()
                    if mass_less_zero > p_mass or mass_less_zero < (1 - p_mass):
                        ft_name = ft.replace("influence_", "")
                        ft_name = ft_name.replace("active_", "")
                        significant_ft.append(ft_name)
                else:
                    # single variabel with shape for all ft_inter_names
                    masses_less_zero = (lin_trace[ft] > 0).mean(axis=0)
                    relevant_mask = np.any(
                        [masses_less_zero > p_mass, masses_less_zero < (1 - p_mass)],
                        axis=0,
                    )
                    rel_idx = np.nonzero(relevant_mask)
                    significant_ft = np.array(ft_inter_names)[rel_idx]
        return significant_ft

    @staticmethod
    def calc_confidence_err(conf_prob, y_eval, y_trace):
        y_conf = az.hdi(y_trace, credible_interval=conf_prob)

        pred_in_conf_rande_arr = [
            y_low < true_y < y_up for true_y, (y_low, y_up) in zip(y_eval, y_conf)
        ]
        in_range_ratio = np.array(pred_in_conf_rande_arr).mean()
        return in_range_ratio

    @staticmethod
    def calc_confidence_closest_mape(conf_prob, y_eval, y_trace):
        y_conf = az.hdi(y_trace, credible_interval=conf_prob)
        closest_mape = []
        for true_y, (y_low, y_up) in zip(y_eval, y_conf):
            if y_low <= true_y <= y_up:
                mape = 0
            elif y_low > true_y:
                mape = (y_low - true_y) / true_y
            else:
                mape = (true_y - y_up) / true_y
            mape *= 100  # make percent
            closest_mape.append(mape)
        np_mapes = np.array(closest_mape)
        closest_mape = float(np.array(np_mapes).mean())
        closest_mape_if_outside = float(np_mapes[np.nonzero(np_mapes)].mean())
        return closest_mape, closest_mape_if_outside

    def predict_2(self, X):
        X_ = self.transform_data_to_candidate_features(self.top_candidate, X)
        pred = self.top_candidate_lr.predict(X_)
        return pred

    def transform_data_to_candidate_features(self, candidate, train_x):
        mapped_features = []
        for term in candidate:
            idx = [self.pos_map[ft] for ft in term]
            selected_cols = np.array(train_x)[:, idx]
            if len(idx) > 1:
                mapped_feature = np.product(selected_cols, axis=1).ravel()
            else:
                mapped_feature = selected_cols.ravel()
            mapped_features.append(list(mapped_feature))
        reshaped_mapped_x = np.atleast_2d(mapped_features).T
        return reshaped_mapped_x

    def fit_and_eval_lin_reg(self, lin_reg_features, reg_proto=None, verbose=True):
        if not reg_proto:
            reg_proto = Ridge()
        inters = [
            get_feature_names_from_rv_id(ft_inter_Str)
            for ft_inter_Str in lin_reg_features
        ]
        x_mapped = self.transform_data_to_candidate_features(inters, self.X)
        lr = copy.deepcopy(reg_proto)
        lr.fit(x_mapped, self.y)
        if verbose:
            print_scores("analogue LR", lr, "train set", x_mapped, self.y)
        errs = get_err_dict(lr, x_mapped, self.y)
        return lr, errs

    def get_p4_train_data(self, X=None):
        vars_and_biases = self.final_var_names
        rv_names = []
        inter_strs = []
        columns = []
        for var, _ in vars_and_biases.items():
            if len(var) == 1:
                var_name = var[0]
                idx_ft = self.pos_map[var_name]
                vals_ft_np = X[:, idx_ft]
                columns.append(vals_ft_np)
                rv_names.append(var_name)
                inter_str = "influence_{}".format(var_name)
                inter_strs.append(inter_str)
            else:
                a, b = var
                idx_a = self.pos_map[a]
                idx_b = self.pos_map[b]
                inter_combi_str = "{}&{}".format(a, b)
                vals_a_np = X[:, idx_a]
                vals_b_np = X[:, idx_b]
                vals_prod = self.vector_inner_prod_slow(vals_a_np, vals_b_np)

                columns.append(vals_prod)
                rv_names.append(inter_combi_str)
                inter_str = "influence_{}".format(inter_combi_str)
                inter_strs.append(inter_str)
        train_data = np.concatenate([c.reshape((-1, 1)) for c in columns], axis=1)
        return rv_names, train_data

    def save_spectrum_fig(self, reg_dict_final, err_dict, rv_names):
        iteration_id = iteration_id = hash(tuple(rv_names))
        print("saving spectrum")
        err_tuples = [
            (i, errs["r2"], errs["mape"], errs["rmse"])
            for i, errs in enumerate(list(err_dict.values()))
        ]
        df_errors = pd.DataFrame(err_tuples, columns=["i", "r2", "mape", "rmse"])
        df_melted_errs = pd.melt(
            df_errors,
            id_vars=["i"],
            value_vars=["r2", "mape", "rmse"],
            var_name="error type",
            value_name="model error",
        )
        g = sns.FacetGrid(df_melted_errs, col="error type", sharex=False, sharey=False)
        g = g.map(plt.hist, "model error")
        spectrum_id = "spect-{}".format(iteration_id)
        err_id = "{}-errs".format(spectrum_id)
        self.saver.store_figure(err_id)
        coef_list = [
            (i, reg.intercept_, *list(reg.coef_))
            for i, reg in enumerate(reg_dict_final.values())
        ]
        coef_cols = ["root", *rv_names]
        df_coefs = pd.DataFrame(coef_list, columns=["i", *coef_cols])
        df_melted_coefs = pd.melt(
            df_coefs,
            id_vars=["i"],
            value_vars=coef_cols,
            var_name="feature",
            value_name="value",
        )
        g = sns.FacetGrid(
            df_melted_coefs, col="feature", sharex=False, sharey=False, col_wrap=4
        )

        g = g.map(sns.histplot, "value")
        coef_id = "{}-coefs".format(spectrum_id)
        self.saver.store_figure(coef_id)
        pass

    # def fit(
    #     self,
    #     X,
    #     y,
    #     feature_names=None,
    #     pos_map=None,
    #     attribute="unknown-attrib",
    #     mcmc_tune=2000,
    #     mcmc_cores=3,
    #     mcmc_samples=1000,
    #     model_interactions=True,
    # ):
    #     fit_start = time.time()
    #     self.X = np.array(X)
    #     self.y = np.array(y)
    #     if feature_names:
    #         self.feature_names = feature_names
    #     if pos_map:
    #         self.pos_map = pos_map
    #     else:
    #         self.pos_map = {opt: idx for idx, opt in enumerate(feature_names)}
    #
    #     if model_interactions:
    #         if self.t_wise:
    #             self.interactions_possible = self.t_wise > 1
    #             print("Interactions possible because t =", self.t_wise)
    #         else:
    #             self.interactions_possible = len(self.X[0]) < len(self.X)
    #         noise_str = "noise"
    #     else:
    #         self.interactions_possible = False
    #
    #     self.x_shared = np.array(self.X)
    #
    #     start_ft_selection = time.time()
    #     print("Starting feature and interaction selection.")
    #     self.final_var_names, lars_pipe, pruned_x = self.get_influentials_from_lasso()
    #     assert (
    #         self.final_var_names
    #     ), "Lasso feature selection selected no options of interactions. Hence, P4 cannot learn any influence!"
    #     self.cost_ft_selection = time.time() - start_ft_selection
    #     print("Feature selection with lasso took {}s".format(self.cost_ft_selection))
    #     stage_name_lasso = "it-lasso"
    #     print_flush("Starting {}".format(stage_name_lasso))
    #     lasso_start = time.time()
    #     lasso_model, lasso_reg_features, lasso_trace = self.get_and_fit_model_biased(
    #         mcmc_cores, mcmc_samples, mcmc_tune
    #     )
    #
    #     lasso_end = time.time()
    #     self.fitting_times[stage_name_lasso] = lasso_end - lasso_start
    #     stage_2_influential_fts = [f for f in self.final_var_names if len(f) == 1]
    #     stage_2_influential_inters = [f for f in self.final_var_names if len(f) > 1]
    #     # stage_2_influential_fts, stage_2_influential_inters = self.get_ft_and_inters_from_rvs(lasso_trace, noise_str,
    #     #                                                                                       p_mass,
    #     #                                                                                       lasso_reg_features)
    #     print("Starting snapshot")
    #     snapshot = self.construct_snapshot(
    #         stage_2_influential_fts,
    #         stage_2_influential_inters,
    #         lasso_model,
    #         lasso_reg_features,
    #         lasso_trace,
    #         stage_name=stage_name_lasso,
    #     )
    #     self.history[stage_name_lasso] = snapshot
    #     model_trace_dict = get_model_trace_dict(lasso_trace, lasso_model)
    #     self.models["it-final"] = model_trace_dict
    #     print_flush("Finished snapshot")
    #     self.final_model = lasso_model
    #     self.final_trace = lasso_trace
    #     print("linked trace to Tracer object")
    #     fit_end = time.time()
    #     self.total_experiment_time = fit_end - fit_start

    def predict_raw_keep_trace_samples(
        self, x, model=None, trace=None, n_post_samples=None
    ):

        transformed_x = self.get_p4_train_data(np.array(x))[1]
        return prediction_samples

    def get_influentials_from_lasso(self, X, y, degree=2):
        train_x_2d = np.atleast_2d(X)
        train_y = y
        lars = LassoCV(
            cv=3,
            positive=False,
            max_iter=5000,
        )  # .fit(train_x_2d, train_y)
        if self.interactions_possible:
            poly_mapping = PolynomialFeatures(
                degree, interaction_only=True, include_bias=False
            )
            lars_pipe = make_pipeline(poly_mapping, lars)
        else:
            lars_pipe = lars
        lars_pipe.fit(train_x_2d, train_y)

        if self.interactions_possible:
            transformed_x = poly_mapping.transform(train_x_2d)
            if self.feature_names is not None:
                ft_inters = poly_mapping.get_feature_names_out(
                    input_features=self.feature_names
                )
            else:
                ft_inters = poly_mapping.get_feature_names_out()
        else:
            transformed_x = train_x_2d
            ft_inters = self.feature_names
        coefs = lars.coef_

        ft_inters_and_influences = {}
        inf_idx = []
        for i, (c, ft_inter) in enumerate(zip(coefs, ft_inters)):
            if c != 0.0:
                inf_idx.append(i)
                ft_inters_and_influences[tuple(ft_inter.split())] = c
        pruned_x = transformed_x[:, inf_idx]

        ft_inters_and_influences = {
            tuple(ft_inter.split()): c
            for c, ft_inter in zip(coefs, ft_inters)
            if c != 0.0
        }
        return ft_inters_and_influences, lars_pipe, pruned_x


def get_feature_names_from_rv_id(ft_inter):
    new_ft_inter = ft_inter.replace("_log__", "")
    new_ft_inter = new_ft_inter.replace("active_", "")
    new_ft_inter = new_ft_inter.replace("_scale", "")
    new_ft_inter = new_ft_inter.replace("influence_", "")
    result = new_ft_inter.split("&")
    return result


def print_baseline_perf(train_x, train_y, eval_x, eval_y):
    train_x_2d = np.atleast_2d(train_x)
    rf = RandomForestRegressor(n_estimators=10, n_jobs=1).fit(train_x_2d, train_y)
    lr = LinearRegression(n_jobs=1).fit(train_x_2d, train_y)
    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(train_x_2d, train_y)
    lars = LassoLars(alpha=0.1, positive=False, fit_path=True).fit(train_x_2d, train_y)

    for name, reg in [
        ("lr", lr),
        ("rf", rf),
        ("ridge", ridge),
        ("lars", lars),
    ]:
        print("\t__ {} __".format((name)))
        print_scores(name, reg, "train", train_x, train_y)
        print_scores(name, reg, "eval", eval_x, eval_y)
        print("")

    print("lr intercept: {}".format(lr.intercept_))
    print("lr coefs: {}".format(lr.coef_))
    print("ridge intercept: {}".format(ridge.intercept_))
    print("ridge coefs: {}".format(ridge.coef_))


def get_random_seed():
    return np.random.randint(np.iinfo(np.uint32).max)


def print_scores(model_name, reg, sample_set_id, xs, ys, print_raw=False):
    errors = get_err_dict(reg, xs, ys)
    for score_id, score in errors.items():
        if not print_raw and "raw" in score_id:
            continue
        print(
            "{} {} set {} score: {}".format(model_name, sample_set_id, score_id, score)
        )
    print()


def get_err_dict(reg, xs, ys):
    y_pred = reg.predict(xs)
    errors = get_err_dict_from_predictions(y_pred, xs, ys)
    return errors


def get_err_dict_from_predictions(y_pred, xs, ys):
    mape = score_mape(None, xs, ys, y_pred)
    rmse = score_rmse(None, xs, ys, y_pred)
    r2 = r2_score(ys, y_pred)
    errors = {
        "r2": r2,
        "mape": mape,
        "rmse": rmse,
        "raw": {"x": xs, "y_pred": y_pred, "y_true": ys},
    }
    return errors


def score_rmse(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    rms = sqrt(mean_squared_error(y_true, y_predicted))
    return rms


def score_mape(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    mape = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
    return mape


def get_snapshot_dict(
    lr_reg_dict,
    err_dict,
    errors,
    lin_reg_features,
    linear_model,
    significant_ft,
    trace,
    trace_errs,
):
    snapshot = {
        "used-features": lin_reg_features,
        "significant-ft": significant_ft,
        "prob-err": errors,
        "prob-model": linear_model,
        "lr-err": err_dict,
        "lr-models": lr_reg_dict,
        "trace": trace,
        "pred-trace-errs": trace_errs,
    }
    return snapshot


def remove_raw_field(inter_errors):
    inter_errors = {key: val for key, val in inter_errors.items() if key != "raw"}
    return inter_errors


def get_model_trace_dict(lin_trace, linear_model):
    model_trace_dict = {"model": linear_model, "trace": lin_trace}
    return model_trace_dict


def print_flush(print_text):
    print(print_text)
    sys.stdout.flush()


def weighted_avg_and_std(values, weights, gamma=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if gamma != 1:
        weights = np.power(weights, gamma)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    if variance <= 0:
        sqr_var = 0.0
    else:
        sqr_var = math.sqrt(variance)
    return average, sqr_var


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


class PyroMCMCRegressor:
    def __init__(
        self,
        mcmc_samples: int = 1000,
        mcmc_tune=1000,
        n_chains=1,
    ):
        self.error_prior = None
        self.infl_prior = None
        self.base_prior = None
        self.coef_ = None
        self.samples = None
        # self.grammar = grammar
        self.mcmc_samples = mcmc_samples
        self.mcmc_tune = mcmc_tune
        self.n_chains = n_chains
        self.mcmc = None

    def model(
        self,
        data,
        y=None,
        base_prior=None,
        infl_prior=None,
        error_prior=None,
    ):
        base_prior = self.base_prior if base_prior is None else base_prior
        self.base_prior = base_prior
        infl_prior = self.infl_prior if infl_prior is None else infl_prior
        self.infl_prior = infl_prior
        error_prior = self.error_prior if error_prior is None else error_prior
        self.error_prior = error_prior
        if y is not None:
            y = jnp.array(y)
        data = jnp.array(data)
        base = numpyro.sample(
            "base",
            # dist.Normal(0, 1)  # dist.Normal(self.prior_root_mean, self.prior_root_std)
            base_prior
            # "base",
            # base_prior,
        )
        rnd_influences = numpyro.sample(
            "coefs",
            infl_prior
            # dist.Normal(jnp.zeros(data.shape[1]), jnp.ones(data.shape[1]))
            # dist.Normal(
            #     self.prior_coef_means,
            #     self.prior_coef_stdvs,
            # ),
        )
        mat_infl = rnd_influences.reshape(-1, 1)
        product = jnp.matmul(data, mat_infl).reshape(-1)
        result = product + base
        error_var = numpyro.sample(
            # "error", dist.Gamma(self.gamma_alpha, self.gamma_beta)
            "error",
            # dist.Exponential(1)
            error_prior,
        )
        with numpyro.plate("data_vectorized", len(result)):
            obs = numpyro.sample("measurements", dist.Normal(result, error_var), obs=y)
        return obs

    def fit(
        self,
        X,
        y,
        random_key=0,
        verbose=False,
        feature_names=None,
        pos_map=None,
        mcmc_tune=None,
        mcmc_cores=None,
        mcmc_samples=None,
    ):
        self.rv_names = (
            get_n_words(X.shape[1])
            if feature_names is None
            else ["&".join(option for option in feature) for feature in feature_names]
        )

        (
            coef_prior,
            base_prior,
            error_prior,
            self.weighted_errs_per_sample,
            self.weighted_rel_errs_per_sample,
        ) = self.get_prior_weighted_normal(X, y, self.rv_names, gamma=3)

        rng_key = random.PRNGKey(random_key)
        nuts_kernel = NUTS(self.model, adapt_step_size=True)
        n_samples = mcmc_samples if mcmc_samples else self.mcmc_samples
        n_tune = mcmc_tune if mcmc_tune else self.mcmc_tune
        n_chains = mcmc_cores if mcmc_cores else self.n_chains
        mcmc = MCMC(
            nuts_kernel,
            num_samples=n_samples,
            num_warmup=n_tune,
            num_chains=n_chains,
        )
        mcmc.run(
            rng_key,
            X,
            y,
            base_prior=base_prior,
            infl_prior=coef_prior,
            error_prior=error_prior,
        )
        self.samples = mcmc.get_samples()
        if verbose:
            pprint(self.samples)
            mcmc.print_summary()
        self.mcmc = mcmc
        self.update_coefs()

    def update_coefs(self):
        """
        Uses the current inferred trace to compute self.coef_ and self.coef_samples_
        """
        root_samples = np.array(self.samples["base"])
        influence_samples = np.array(self.samples["coefs"])
        influence_dict = {
            varname: np.array(influence_samples[:, i])
            for i, varname in enumerate(self.rv_names)
        }
        relative_error_samples = np.array(self.samples["base"])
        self.coef_samples_ = {
            "root": root_samples,
            "influences": influence_dict,
            "relative_error": relative_error_samples,
        }
        root_mode = float(np.mean(az.hdi(root_samples, hdi_prob=0.01)))
        influence_modes = list(
            np.mean(az.hdi(influence_samples, hdi_prob=0.01), axis=1)
        )
        influence_modes_dict = {
            varname: mode for mode, varname in zip(influence_modes, self.rv_names)
        }
        rel_error_mode = float(np.mean(az.hdi(relative_error_samples, hdi_prob=0.01)))
        self.coef_ = {
            "root": root_mode,
            "influences": influence_modes_dict,
            "relative_error": rel_error_mode,
        }

    def get_prior_weighted_normal(self, X, y, rv_names, gamma=1, stddev_multiplier=3):
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(X, y, rv_names)
        all_raw_errs = [errs["raw"] for errs in list(err_dict.values())]
        all_abs_errs = np.array(
            [abs(err["y_pred"] - err["y_true"]) for err in all_raw_errs]
        )
        mean_abs_errs = all_abs_errs.mean(axis=1)
        all_rel_errs = np.array(
            [
                abs((err["y_pred"] - err["y_true"]) / err["y_true"])
                for err in all_raw_errs
            ]
        )
        mean_rel_errs = all_rel_errs.mean(axis=1)
        reg_list = list(reg_dict_final.values())

        means_weighted = []
        stds_weighted = []
        weights = (
            1 - MinMaxScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        )
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=gamma)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        root_candidates = np.array([reg.intercept_ for reg in reg_list])
        root_mean, root_std = weighted_avg_and_std(
            root_candidates, weights, gamma=gamma
        )
        for coef_id, coef in enumerate(rv_names):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list])
            mean_weighted, std_weighted = weighted_avg_and_std(
                coef_candidates, weights, gamma=gamma
            )
            means_weighted.append(mean_weighted)
            stds_weighted.append(stddev_multiplier * std_weighted)

        weighted_errs_per_sample = np.average(
            all_abs_errs, axis=0, weights=mean_abs_errs
        )
        weighted_rel_errs_per_sample = np.average(
            all_rel_errs, axis=0, weights=mean_rel_errs
        )

        base_prior = dist.Normal(jnp.array(root_mean), jnp.array(root_std))
        error_prior = dist.Exponential(jnp.array(err_mean))
        coef_prior = dist.Normal(
            jnp.array(means_weighted),
            jnp.array(stds_weighted),
        )

        return (
            coef_prior,
            base_prior,
            error_prior,
            weighted_errs_per_sample,
            weighted_rel_errs_per_sample,
        )

    def get_regression_spectrum(
        self, X, y, lin_reg_features, n_steps=50, cv=3, n_jobs=-1
    ):
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
        for l1_ratio in step_list:
            if 0 < l1_ratio < 1:
                reg_prototype = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, n_jobs=n_jobs)
                reg, err = self.fit_and_eval_lin_reg(
                    X, y, lin_reg_features, reg_proto=reg_prototype, verbose=False
                )
                regs.append((reg, err))
        ridge = RidgeCV(cv=cv)
        lasso = LassoCV(cv=cv, n_jobs=n_jobs)
        for reg in [ridge, lasso]:
            fitted_reg, err = self.fit_and_eval_lin_reg(
                X, y, lin_reg_features, reg_proto=reg, verbose=False
            )
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        self.prior_spectrum_cost = cost
        print("Prior Spectrum Computation took", cost)

        return reg_dict, err_dict

    def fit_and_eval_lin_reg(
        self, X, y, lin_reg_features, reg_proto=None, verbose=True
    ):
        if not reg_proto:
            reg_proto = Ridge()
        inters = [
            get_feature_names_from_rv_id(ft_inter_Str)
            for ft_inter_Str in lin_reg_features
        ]
        x_mapped = X  # self.transform_data_to_candidate_features(inters, X)
        lr = copy.deepcopy(reg_proto)
        lr.fit(x_mapped, y)
        if verbose:
            print_scores("analogue LR", lr, "train set", x_mapped, y)
        errs = get_err_dict(lr, x_mapped, y)
        return lr, errs

    def get_tuples(self, feature_names):
        tuples = []
        tuples.extend(
            [("mcmc", "base", float(val)) for val in self.samples["base"].numpy()]
        )
        for n, rv_name in enumerate(feature_names):
            rv_samples = self.samples["coefs"][:, n]
            tuples.extend([("mcmc", rv_name, float(val)) for val in rv_samples.numpy()])
        return tuples

    def _predict_samples(self, X, n_samples: int = None, rnd_key=0):
        # Predictive(model_fast, guide=guide, num_samples=100,
        # return_sites=("measurements",))
        n = n_samples if n_samples else self.samples
        prngkey = random.PRNGKey(rnd_key)
        pred = Predictive(self.model, num_samples=n)
        posterior_samples = pred(prngkey, X, None)
        y_pred = posterior_samples["measurements"]
        return y_pred

    def predict(self, X, n_samples: int = None, ci: float = None):
        """
        Performs a prediction conforming to the sklearn interface.

        Parameters
        ----------
        X : Array-like data
        n_samples : number of posterior predictive samples to return for each prediction
        ci : value between 0 and 1 representing the desired confidence of returned confidence intervals. E.g., ci= 0.8 will generate 80%-confidence intervals

        Returns
        -------
         - a scalar if only x is specified
         - a set of posterior predictive samples of size n_samples if is given and n_samples > 0
         - a set of pairs, representing lower and upper bounds of confidence intervals for each prediction if ci is given

        """
        if not n_samples:
            n_samples = 500
            y_samples = self._predict_samples(X, n_samples=n_samples)
            y_pred = np.mean(az.hdi(y_samples, hdi_prob=0.01), axis=1)
        else:
            y_samples = self._predict_samples(X, n_samples=n_samples)
            if ci:
                assert_ci(ci)
                y_pred = az.hdi(y_samples, hdi_prob=ci)
            else:
                y_pred = y_samples
        return y_pred

    def coef_ci(self, ci: float):
        """
        Returns confidence intervals with custom confidence for p
        Parameters
        ----------
        ci : specifies confidence of the confidence interval to compute from sampled influence values

        Returns dictionary containing a confidence interval for each influence
        -------

        """
        coef_cis = {}
        assert_ci(ci)
        for key, val in self.coef_samples_.items():
            if key == "influences":
                inf_dict = {}
                for feature_name, feature_samples in val.items():
                    feature_name = tuple(get_feature_names_from_rv_id(feature_name))
                    inf_dict[feature_name] = az.hdi(feature_samples, hdi_prob=ci)
                coef_cis[key] = inf_dict
            else:
                coef_cis[key] = az.hdi(val, hdi_prob=ci)
        return coef_cis

    def fit_pm_model(
        self, mcmc_cores, mcmc_samples, mcmc_tune, rv_names, train_data, observed_y
    ):
        (
            prior_coef_means,
            prior_coef_stdvs,
            prior_root_mean,
            prior_root_std,
            err_mean,
            err_std,
            self.weighted_errs_per_sample,
            self.weighted_rel_errs_per_sample,
        ) = self.get_prior_weighted_normal(rv_names, gamma=3)
        gamma_prior = gamma.fit(
            self.weighted_errs_per_sample,
        )
        gamma_shape, gamma_loc, gamma_scale = gamma_prior
        gamma_k = gamma_shape
        gamma_theta = gamma_scale
        gamma_alpha = gamma_k
        gamma_beta = 1 / gamma_theta
        # rel_err = jnp.mean(self.weighted_rel_errs_per_sample)
        pyro_reg = PyroMCMCRegressor(
            mcmc_samples,
            mcmc_tune,
            mcmc_cores,
            prior_root_mean,
            prior_root_std,
            prior_coef_means,
            prior_coef_stdvs,
            gamma_alpha,
            gamma_beta,
        )
        start = time.time()
        # storing_start = time.time()
        # print("Storing Prior Model pickle")
        # self.saver.store_pickle(pyro_model, "prior-model")
        # print("Successfully stored Prior Model pickle")
        # storing_cost = time.time() - storing_start
        storing_cost = 0
        pyro_reg.fit(train_data, observed_y, self.rnd_seed)
        self.samples = pyro_reg.samples
        end = time.time()
        print_flush("Done Fitting. Calulating time.")
        total_cost = (end - start) - storing_cost
        print_flush("Fitted model in {0:9.1} minutes.".format((total_cost) / 60))
        return pyro_reg, self.samples

    def get_arviz_dims(self):
        dims = {
            "coefs": ["features"],
        }
        return dims

    def get_arviz_data(
        self,
    ):
        coords = {
            "features": self.rv_names,
        }
        dims = self.get_arviz_dims()
        idata_kwargs = {
            "dims": dims,
            "coords": coords,
        }
        az_data = az.from_numpyro(self.mcmc, num_chains=self.n_chains, **idata_kwargs)

        return az_data

    def loo(self, pointwise=False, scale="log"):
        """
        Returns the PSIS information criterion. Used to compare models.
        If using "log" scale, higher is better. Some literature uses "deviance", which equals -2 * log-scale-PSIS.
        If pointwise=True, returns the posterior log likelihood of each training data point.
        Returns the aggregated PSIS score (sum of the log likelihoods).
        """
        az_data = self.get_arviz_data()
        psis = az.loo(az_data, pointwise=pointwise, scale=scale)
        if pointwise:
            elpd_psis = psis.loo_i
        else:
            elpd_psis = psis.loo
        return elpd_psis


def assert_ci(ci):
    assert 0 < ci < 1, "Confidence should be given 0 < ci < 1"


class SaverHelper:
    def __init__(self, path, dpi=800, fig_pre="fig"):
        self.path = path
        self.session_dir = None
        self.dpi = dpi
        self.fig_pre = fig_pre

    @staticmethod
    def get_clean_file_name(f_name):
        f_name_clean = f_name.replace(" ", "")
        return f_name_clean

    def store_xml(self, xml_root, f_name, folder="."):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_xml = os.path.join(current_folder, "run-conf-{}.xml".format(f_name_clean))
        # xml_string = pprint.pformat(xml_root)
        xml_string = ET.tostring(xml_root).decode("utf-8")
        xml_string = "\n".join([x for x in xml_string.split("\n") if x.strip() != ""])

        with open(file_xml, "w") as f:
            f.write(xml_string)
        abs_path = os.path.abspath(file_xml)
        return abs_path

    def store_dict(self, r_dict, f_name, folder="."):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_pickle = os.path.join(current_folder, "results-{}.p".format(f_name_clean))
        file_txt = os.path.join(current_folder, "results-{}.txt".format(f_name_clean))
        with open(file_pickle, "wb") as f:
            pickle.dump(r_dict, f)
        dict_string = pformat(r_dict)
        with open(file_txt, "w") as f:
            f.write(dict_string)
        abs_path = os.path.abspath(file_pickle)
        return abs_path

    def store_pickle(self, obj, f_name, folder="."):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        file_pickle = os.path.join(current_folder, "results-{}.p".format(f_name_clean))
        with open(file_pickle, "wb") as f:
            pickle.dump(obj, f)
        abs_path = os.path.abspath(file_pickle)
        return abs_path

    def store_figure(self, f_name_clean, folder="."):
        f_name_clean = self.get_clean_file_name(f_name_clean)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        for extension in ("pdf", "png"):
            f_name_final = "{}-{}.{}".format(self.fig_pre, f_name_clean, extension)
            f_path = os.path.join(current_folder, f_name_final)

            plt.savefig(f_path, dpi=self.dpi)

    def set_session_dir(self, name):
        self.session_dir = name
        return self.get_cwd()

    def safe_folder_join(self, *args):
        path = os.path.join(*args)
        os.makedirs(path, exist_ok=True)
        return path

    def get_cwd(self):
        cwd = os.path.join(self.path, self.session_dir)
        return cwd


def get_time_str(i=None):
    i = datetime.datetime.now() if i is None else datetime.datetime.fromtimestamp(i)
    time_str = "-".join(
        (str(intt) for intt in [i.year, i.month, i.day, i.hour, i.minute, i.second])
    )
    return time_str
