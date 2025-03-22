import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from joblib import Parallel, delayed
from conditionalconformal import CondConf
from localized_conformal_utils import *
import bisect

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Conformal_UQ:
    def __init__(self, model):
        """
        Initializes the Conformal_UQ_roc class.

        Parameters:
        ----------
        model : object
            A machine learning model that implements `fit` and `predict` methods.
        alphas : np.ndarray, optional
            Array of significance levels (1 - confidence levels) for prediction intervals.
        """
        self.model = clone(model)
        self.trained = False
        self.calibrated = False
        self.predicted = False
        
    def train_predict(self, x_train, y_train, 
                      x_val, y_val, x_test,
                      alpha=0.1):
        
        _ = self.train(x_train = x_train, y_train = y_train)
        _ = self.calibrate(x_val = x_val, y_val = y_val, alpha = alpha)
        pred_intervals = self.predict(x_test = x_test)

        return pred_intervals
    

    def train(self, x_train, y_train):
        """
        Fits the model on training data and computes calibration errors on validation data.

        Parameters:
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        x_val : np.ndarray
        y_val : np.ndarray
        """
        # Train the model
        self.model.fit(x_train, y_train)
        self.trained = True
    
    def calibrate(self, x_val, y_val, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        
        self.alpha = alpha

        # Predict on the validation set
        y_calib_pred = self.model.predict(x_val)

        # Compute nonconformity scores (absolute residuals)
        self.calib_errors = np.abs(y_val - y_calib_pred)

        # Compute quantile for the specified alpha
        n2 = len(self.calib_errors)
        self.q = np.sort(self.calib_errors)[int(np.ceil((n2 + 1) * (1 - self.alpha)) - 1)]

        self.calibrated = True
        return self.q

    def predict(self, x_test):
        """
        Predicts intervals for test data using calibration errors.

        Parameters:
        ----------
        x_test : np.ndarray
            Test features.

        Returns:
        -------
        list of pd.DataFrame
            A list of DataFrames containing prediction intervals for each alpha level.
        """
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")

        # Predict on the test set
        y_pred_test = self.model.predict(x_test)

        # Compute prediction intervals
        lower_bounds = y_pred_test - self.q
        upper_bounds = y_pred_test + self.q

        # Create a DataFrame for the prediction intervals
        self.prediction_intervals = pd.DataFrame({
            "lb": lower_bounds,
            "ub": upper_bounds,
            "point_est": y_pred_test,
            "alpha": self.alpha
        })
        self.predicted = True

        return self.prediction_intervals

    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")

        lower_bounds = self.prediction_intervals["lb"]
        upper_bounds = self.prediction_intervals["ub"]

        coverage = np.mean((lower_bounds <= y_test) & (y_test <= upper_bounds))
        avg_length = np.mean(upper_bounds - lower_bounds)
        med_length = np.median(upper_bounds - lower_bounds)
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])
        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df
        

    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified["subgroup"] = subgroups

        lower_bounds = pred_modified["lb"]
        upper_bounds = pred_modified["ub"]

        pred_modified['width'] = upper_bounds - lower_bounds
        pred_modified['covers'] = (lower_bounds <= pred_modified['truth']) & (pred_modified['truth'] <= upper_bounds)


        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("width", "mean"),
            median_length = ("width", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics

class Conformal_Locally_Weighted:
    def __init__(self, mean_model, sd_model):
        self.mean_model = clone(mean_model)
        self.sd_model = clone(sd_model)
        self.trianed = False
        self.calibrated = False
        self.predicted = False

    def train_predict(self, x_train, y_train,
                        x_val, y_val, x_test,
                        abs_error=True, alpha=0.1):
        _ = self.train(x_train = x_train, y_train = y_train, abs_error = abs_error)
        _ = self.calibrate(x_val = x_val, y_val = y_val, alpha = alpha)
        pred_intervals = self.predict(x_test = x_test)

        return pred_intervals
    
    def train(self, x_train, y_train, abs_error = True):
        self.mean_model.fit(x_train, y_train)
        # train mean model on training set
        self.mean_model.fit(x_train, y_train)

        # get training errors
        train_pred_mean = self.mean_model.predict(x_train)
        train_errors = y_train - train_pred_mean

        # train sd moddel on training set with training errors
        if abs_error:
            self.sd_model.fit(x_train, np.abs(train_errors))
        else:
            self.sd_model.fit(x_train, train_errors)
        
        self.trained = True
    
    def calibrate(self, x_val, y_val, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        self.alpha = alpha
        # get validation errors
        val_pred_mean = self.mean_model.predict(x_val)
        val_errors = y_val - val_pred_mean
        val_abs_errors = np.abs(val_errors)

        # predict validation sd
        val_pred_error = self.sd_model.predict(x_val)
        val_pred_sd = np.abs(val_pred_error)

        # weigh the errors
        weighted_errors_val = val_abs_errors / val_pred_sd

        # get quantile
        n2 = len(weighted_errors_val)
        self.q = np.sort(weighted_errors_val)[int(np.ceil((n2 + 1) * (1 - alpha)) - 1)]

        self.calibrated = True
        return self.q
    
    def predict(self, x_test):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        
        # predict on test set
        test_pred_mean = self.mean_model.predict(x_test)
        test_pred_error = self.sd_model.predict(x_test)
        test_pred_sd = np.abs(test_pred_error)
        
        # make interval
        lb = test_pred_mean - self.q * test_pred_sd
        ub = test_pred_mean + self.q * test_pred_sd
        
        self.prediction_intervals = pd.DataFrame({"lb": lb, "ub": ub, "point_est": test_pred_mean})
        
        self.predicted = True
        return self.prediction_intervals
    
    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")

        lower_bounds = self.prediction_intervals["lb"]
        upper_bounds = self.prediction_intervals["ub"]

        coverage = np.mean((lower_bounds <= y_test) & (y_test <= upper_bounds))
        avg_length = np.mean(upper_bounds - lower_bounds)
        med_length = np.median(upper_bounds - lower_bounds)
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])
        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df

    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified["subgroup"] = subgroups

        lower_bounds = pred_modified["lb"]
        upper_bounds = pred_modified["ub"]

        pred_modified['width'] = upper_bounds - lower_bounds
        pred_modified['covers'] = (lower_bounds <= pred_modified['truth']) & (pred_modified['truth'] <= upper_bounds)


        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("width", "mean"),
            median_length = ("width", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics
    
class Conformal_Quantiles:
    def __init__(self, model):
        self.model = model
        self.trained = False
        self.calibrated = False
        self.predicted = False
    
    def train_predict(self, x_train, y_train,
                        x_val, y_val, x_test,
                        alpha=0.1):
        _ = self.train(x_train = x_train, y_train = y_train, alpha = alpha)
        _ = self.calibrate(x_val = x_val, y_val = y_val)
        pred_intervals = self.predict(x_test = x_test)

        return pred_intervals

    def train(self, x_train, y_train, alpha=0.1):
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = alpha
        self.lower_quantile = self.alpha / 2
        self.upper_quantile = 1 - self.alpha / 2

        if isinstance(self.model, QuantileRegressor):
            self.lower_model = QuantileRegressor(quantile=self.lower_quantile, solver="highs")
            self.upper_model = QuantileRegressor(quantile=self.upper_quantile, solver="highs")

            self.lower_model.fit(x_train, y_train)
            self.upper_model.fit(x_train, y_train)
        
        elif isinstance(self.model, RandomForestQuantileRegressor):
            # Train a Quantile Forest model for both bounds
            self.model.fit(x_train, y_train)

        else:
            raise ValueError(
                "Unsupported model type. Please use QuantileRegressor or RandomForestQuantileRegressor."
            )
        self.trained = True
    
    def calibrate(self, x_val, y_val):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        
        if isinstance(self.model, QuantileRegressor):
            lower_val = self.lower_model.predict(x_val)
            upper_val = self.upper_model.predict(x_val)

            errors = np.maximum(lower_val - y_val, y_val - upper_val)
            n2 = len(errors)
            self.q = np.sort(errors)[int(np.ceil((n2 + 1) * (1 - self.alpha)) - 1)]

        elif isinstance(self.model, RandomForestQuantileRegressor):
            # Train a Quantile Forest model for both bounds
            lower_val = self.model.predict(x_val, quantiles=self.lower_quantile)
            upper_val = self.model.predict(x_val, quantiles=self.upper_quantile)

            errors = np.maximum(lower_val - y_val, y_val - upper_val)
            n2 = len(errors)
            self.q = np.sort(errors)[int(np.ceil((n2 + 1) * (1 - self.alpha)) - 1)]

        else:
            raise ValueError(
                "Unsupported model type. Please use QuantileRegressor or RandomForestQuantileRegressor."
            )
        self.calibrated = True
        return self.q        

    def predict(self, x_test):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
            
        if isinstance(self.model, QuantileRegressor):
            lower_bounds = self.lower_model.predict(x_test) - self.q
            upper_bounds = self.upper_model.predict(x_test) + self.q

            # For point estimate, use median quantile
            point_model = QuantileRegressor(quantile=0.5, solver="highs")
            point_model.fit(self.x_train, self.y_train)
            y_pred_test = point_model.predict(x_test)

        elif isinstance(self.model, RandomForestQuantileRegressor):
            # Train a Quantile Forest model for both bounds
            lower_bounds = self.model.predict(x_test, quantiles = self.lower_quantile) - self.q
            upper_bounds = self.model.predict(x_test, quantiles = self.upper_quantile) + self.q
            y_pred_test = self.model.predict(x_test, quantiles=0.5)

        else:
            raise ValueError(
                "Unsupported model type. Please use QuantileRegressor or RandomForestQuantileRegressor."
            )

        self.prediction_intervals = pd.DataFrame({
            "lb": lower_bounds,
            "ub": upper_bounds,
            "point_est": y_pred_test
        })

        self.predicted = True
        return self.prediction_intervals
    
    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")

        lower_bounds = self.prediction_intervals["lb"]
        upper_bounds = self.prediction_intervals["ub"]

        coverage = np.mean((lower_bounds <= y_test) & (y_test <= upper_bounds))
        avg_length = np.mean(upper_bounds - lower_bounds)
        med_length = np.median(upper_bounds - lower_bounds)
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])
        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df

    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified["subgroup"] = subgroups

        lower_bounds = pred_modified["lb"]
        upper_bounds = pred_modified["ub"]

        pred_modified['width'] = upper_bounds - lower_bounds
        pred_modified['covers'] = (lower_bounds <= pred_modified['truth']) & (pred_modified['truth'] <= upper_bounds)


        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("width", "mean"),
            median_length = ("width", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics

class Conformal_Majority_Vote:
    def __init__ (self, models):
        self.models = {model_name: clone(model) for model_name, model in models.items()}
        self.K = len(models)
        self.trained = False
        self.calibrated = False
        self.predicted = False

    def train_predict(self, x_train, y_train,
                        x_val, y_val, x_test,
                        alpha=0.1, tau=0.5):
        _ = self.train(x_train = x_train, y_train = y_train)
        _ = self.calibrate(x_val = x_val, y_val = y_val, alpha = alpha)
        pred_intervals = self.predict(x_test = x_test, tau = tau)

        return pred_intervals

    def train(self, x_train, y_train):
        self.conformals = {}
        for model_name, model in self.models.items():
            single_conformal = Conformal_UQ(model)
            single_conformal.train(x_train, y_train)
            self.conformals[model_name] = single_conformal
        self.trained = True
    
    def calibrate(self, x_val, y_val, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        self.alpha = alpha
        qs = {}
        for model_name, conformal in self.conformals.items():
            q = conformal.calibrate(x_val, y_val, alpha/2)
            qs[model_name] = q

        self.calibrated = True
        return qs
    
    def predict(self, x_test, tau = 0.5):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        self.tau = tau
        lb_cols = [f'{model_name}_lb' for model_name in self.conformals.keys()]
        ub_cols = [f'{model_name}_ub' for model_name in self.conformals.keys()]
        all_conformals = pd.DataFrame(np.zeros((x_test.shape[0], 2*self.K)), 
                                      columns=lb_cols + ub_cols)
        for model_name, conformal in self.conformals.items():
            single_conformal_pred = conformal.predict(x_test)
            all_conformals[f'{model_name}_lb'] = single_conformal_pred['lb']
            all_conformals[f'{model_name}_ub'] = single_conformal_pred['ub']
        self.all_conformals = all_conformals
        self.prediction_intervals = pd.DataFrame(all_conformals.apply(lambda row: self.get_majority_vote(row, self.K, self.tau),axis=1),
                                                columns=['intervals'])

        self.predicted = True
        return self.prediction_intervals

    def get_majority_vote(self, row, K, tau):
        lower_bounds = row.iloc[:K].to_numpy()
        upper_bounds = row.iloc[K:].to_numpy()
        q = np.sort(row)
        i = 1
        lower = []
        upper = []
        while i < 2*K:
            cond_i = np.mean((lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
            if i == 10 and row.name == 1:
                print(lower_bounds, upper_bounds, q[i-1], q[i], (lower_bounds <= (q[i-1] + q[i])/ 2) & (upper_bounds >= (q[i-1] + q[i])/ 2))
            if cond_i > tau:
                lower.append(q[i-1])
                j = i
                cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                while (cond_j > tau) and (j < 2*K):
                    j += 1
                    cond_j = np.mean((lower_bounds <= (q[j-1] + q[j])/ 2) & (upper_bounds >= (q[j-1] + q[j])/ 2))
                i = j
                upper.append(q[i-1])
            else:
                i += 1
        if len(lower) != len(upper):
            raise Exception("Length mismatch in lower and upper bounds")
        intervals = [[lower[l], upper[l]] for l in range(len(lower))]
        return intervals

    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        
        intervals = deepcopy(self.prediction_intervals)
        intervals['truth'] = y_test
        disjoint_metrics = intervals.apply(lambda r: self.get_disjoint_metrics(r), axis=1)

        coverage = disjoint_metrics['covers'].mean()
        avg_length = disjoint_metrics['widths'].mean()
        med_length = np.median(disjoint_metrics['widths'])
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])

        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df
    
    def get_disjoint_metrics(self, row):
        tv = row['truth']
        intvl = row['intervals']
        covers = np.any([tv >= i[0] and tv <= i[1] for i in intvl])
        widths = np.sum([i[1] - i[0] for i in intvl])
        return pd.Series({'covers': covers, 'widths': widths})
    
    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified = pred_modified.apply(lambda r: self.get_disjoint_metrics(r), axis=1)
        pred_modified['truth'] = y_test
        pred_modified["subgroup"] = subgroups

        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("widths", "mean"),
            median_length = ("widths", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics
    

class Conformal_Localized:
    def __init__ (self, model):
        self.model = clone(model)
        self.trained = False
        self.calibrated = False
        self.predicted = False

    def train_predict(self, x_train, y_train,
                        x_val, y_val, x_test,
                        alpha=0.1):
        _ = self.train(x_train = x_train, y_train = y_train)
        _ = self.calibrate(x_val = x_val, y_val = y_val, alpha = alpha)
        pred_intervals = self.predict(x_test = x_test)

        return pred_intervals

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

        # For the cross-validation sample we use the training data:
        self.__Vcv = np.abs(y_train).flatten()
        self.__Dcv = cdist(x_train, x_train, metric='euclidean')

        # Set up a grid of localizer bandwidths. (In 1D the code used quantiles of Dcv;
        # here we use the same idea applied to the multivariate distances.)
        max0 = np.max(self.__Dcv) * 2
        min0 = np.quantile(self.__Dcv, 0.01)
        self.__hs = np.exp(np.linspace(np.log(min0), np.log(max0), 20))
        self.trained = True

    def calibrate(self, x_val, y_val, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        self.alpha = alpha
        self.x_val = x_val
        self.y_val = y_val
        self.__eps = np.abs(self.y_val - self.model.predict(self.x_val)).flatten()
        self.__D = cdist(self.x_val, self.x_val, metric='euclidean')

        # For the LCP module we need to order the calibration scores.
        self.__order1 = np.argsort(self.__eps) 
        D_ordered = self.__D[self.__order1][:, self.__order1]
        eps_ordered = self.__eps[self.__order1]

        # make lcp
        self.LCR = LCP(H=D_ordered, V=eps_ordered, h=0.2, alpha=alpha, type="distance")

        # Auto-tuning: here we call the auto-tune function using the training data
        auto_ret = self.LCR.LCP_auto_tune(V0=self.__Vcv, H0=self.__Dcv, 
                                          hs=self.__hs, B=2, delta=self.alpha/2, 
                                          lambda_=1, trace=True)
        self.LCR.h = auto_ret['h']

        # Prepare the calibration/localizer quantities:
        self.LCR.lower_idx()
        self.LCR.cumsum_unnormalized()
        self.calibrated = True

       
    def predict(self, x_test):
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        # do test prediction
        y_test_pred = self.model.predict(x_test)

        # computer distances
        Dnew = cdist(x_test, self.x_val, metric='euclidean')
        DnewT = Dnew.T

        # make predictions
        self.LCR.LCP_construction(Hnew=Dnew[:, self.__order1], HnewT=DnewT[self.__order1, :])
        deltaLCP = self.LCR.band_V
        lb = y_test_pred - deltaLCP
        ub = y_test_pred + deltaLCP

        self.prediction_intervals = pd.DataFrame({"lb": lb, "ub": ub, "point_est": y_test_pred})
        self.predicted = True
        return self.prediction_intervals
    
    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")

        lower_bounds = self.prediction_intervals["lb"]
        upper_bounds = self.prediction_intervals["ub"]

        inf_index = np.where(upper_bounds == np.Inf)[0]
        inf_percent = len(inf_index) / len(upper_bounds)

        y_test = y_test[upper_bounds != np.Inf]
        lower_bounds = lower_bounds[upper_bounds != np.Inf]
        upper_bounds = upper_bounds[upper_bounds != np.Inf]
        

        coverage = np.mean((lower_bounds <= y_test) & (y_test <= upper_bounds))
        avg_length = np.mean(upper_bounds - lower_bounds)
        med_length = np.median(upper_bounds - lower_bounds)
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha,
            'inf_prop': inf_percent
        }])
        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df
    
    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified["subgroup"] = subgroups

        lower_bounds = pred_modified["lb"]
        upper_bounds = pred_modified["ub"]

        pred_modified['width'] = upper_bounds - lower_bounds
        pred_modified['covers'] = (lower_bounds <= pred_modified['truth']) & (pred_modified['truth'] <= upper_bounds)
        pred_modified['inf'] = (upper_bounds == np.Inf)


        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", lambda x: np.mean(x[x != np.Inf])),
            avg_length = ("width", lambda x: np.mean(x[x != np.Inf])),
            median_length = ("width", lambda x: np.median(x[x != np.Inf])),
            range_y_test = ("truth", lambda x: x.max() - x.min()),
            inf_prop = ("inf", "mean")
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics
    
class Conformal_Conditional:
    def __init__(self, model):
        """
        Initializes the Conformal_UQ_roc class.

        Parameters:
        ----------
        model : object
            A machine learning model that implements `fit` and `predict` methods.
        alphas : np.ndarray, optional
            Array of significance levels (1 - confidence levels) for prediction intervals.
        """
        self.model = clone(model)
        self.trained = False
        self.calibrated = False
        self.predicted = False

    def train_predict(self, x_train, y_train,
                        x_val, y_val, x_test,
                        subgroup_func, alpha=0.1):
        
        _ = self.train(x_train = x_train, y_train = y_train, subgroup_func = subgroup_func)
        _ = self.calibrate(x_val = x_val, y_val = y_val, alpha = alpha)
        pred_intervals = self.predict(x_test = x_test)

        return pred_intervals
        

    def train(self, x_train, y_train, subgroup_func):
        """
        Fits the model on training data and computes calibration errors on validation data.

        Parameters:
        ----------
        x_train : np.ndarray
        y_train : np.ndarray
        x_val : np.ndarray
        y_val : np.ndarray
        """
        # Train the model
        self.model.fit(x_train, y_train)

        num_features = x_train.shape[1]
        self.score_fn = lambda x, y: y - self.model.predict(x)
        self.score_inv_ub = lambda s, x: [-np.inf, self.model.predict(x.reshape(int(x.shape[0]/num_features), num_features)) + s]
        self.score_inv_lb = lambda s, x: [self.model.predict(x.reshape(int(x.shape[0]/num_features), num_features)) + s, np.inf]

        self.subgroup_func = subgroup_func
        self.conf = CondConf(self.score_fn, self.subgroup_func)
        self.trained = True
    
    def calibrate(self, x_val, y_val, alpha=0.1):
        if self.trained == False:
            raise ValueError("Model must be trained before calling calibrate.")
        
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.alpha = alpha
        self.conf.setup_problem(x_val, y_val)

        self.calibrated = True
        return None

    def predict(self, x_test):
        """
        Predicts intervals for test data using calibration errors.

        Parameters:
        ----------
        x_test : np.ndarray
            Test features.

        Returns:
        -------
        list of pd.DataFrame
            A list of DataFrames containing prediction intervals for each alpha level.
        """
        if self.calibrated == False:
            raise ValueError("Model must be calibrated before calling predict.")
        
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)        
        
        lbs = []
        ubs = []
        for i in range(len(x_test)):
            try:
                pred_low = self.conf.predict(self.alpha/2, x_test[i, :], self.score_inv_lb, exact=True, randomize=True, threshold=10**(-10))
                pred_high = self.conf.predict(1-self.alpha/2, x_test[i, :], self.score_inv_ub, exact=True, randomize=True, threshold=10**(-10))
            except ValueError:
                pred_low = self.conf.predict(self.alpha/2, x_test[i, :], self.score_inv_lb, exact=False, randomize=True)
                pred_high = self.conf.predict(1-self.alpha/2, x_test[i, :], self.score_inv_ub, exact=False, randomize=True)
            lbs.append(pred_low[0][0])
            ubs.append(pred_high[1][0])

        # Create a DataFrame for the prediction intervals
        self.prediction_intervals = pd.DataFrame({
            "lb": lbs,
            "ub": ubs,
            "point_est": self.model.predict(x_test),
            "alpha": self.alpha
        })
        self.predicted = True

        return self.prediction_intervals

    def evaluate(self, y_test, scale_width=True):
        """
        Evaluates the prediction intervals 
        and computes coverage and length metrics.

        Parameters
        ----------
        y_test : np.ndarray
            The true target values for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics for each alpha.
        """
        if self.predicted is False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")

        lower_bounds = self.prediction_intervals["lb"]
        upper_bounds = self.prediction_intervals["ub"]

        coverage = np.mean((lower_bounds <= y_test) & (y_test <= upper_bounds))
        avg_length = np.mean(upper_bounds - lower_bounds)
        med_length = np.median(upper_bounds - lower_bounds)
        range_y_test = y_test.max() - y_test.min()

        # Compile results into a DataFrame
        results_df = pd.DataFrame([{
            "coverage": coverage,
            "avg_length": avg_length,
            "median_length": med_length,
            "range_y_test": range_y_test,
            "alpha": self.alpha
        }])
        if scale_width:
            results_df["scaled_avg_length"] = avg_length / range_y_test
            results_df["scaled_median_length"] = med_length / range_y_test

        return results_df
        

    def evaluate_subgroups(self, y_test, subgroups, scale_width=True):
        if self.predicted == False:
            raise ValueError("Please call 'predict' first to generate prediction intervals.")
        pred_modified = deepcopy(self.prediction_intervals)
        pred_modified["truth"] = y_test
        pred_modified["subgroup"] = subgroups

        lower_bounds = pred_modified["lb"]
        upper_bounds = pred_modified["ub"]

        pred_modified['width'] = upper_bounds - lower_bounds
        pred_modified['covers'] = (lower_bounds <= pred_modified['truth']) & (pred_modified['truth'] <= upper_bounds)

        group_metrics = pred_modified.groupby("subgroup").agg(
            coverage = ("covers", "mean"),
            avg_length = ("width", "mean"),
            median_length = ("width", "median"),
            range_y_test = ("truth", lambda x: x.max() - x.min())
        ).reset_index()
        group_metrics["alpha"] = self.alpha

        if scale_width:
            range_y_test = y_test.max() - y_test.min()
            group_metrics["scaled_avg_length"] = group_metrics["avg_length"] / range_y_test
            group_metrics["scaled_median_length"] = group_metrics["median_length"] / range_y_test

        return group_metrics
