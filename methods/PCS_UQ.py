import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy
from gamma_algo import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class PCS_UQ:

    """
    PCS_UQ_roc class performs model checking, bootstrapping, and uncertainty quantification
    using scaled prediction sets on multiple models with performance evaluation.

    Parameters
    ----------
    models : dict
        A dictionary where each key corresponds to a group of models, and the values are
        dictionaries containing model names and model instances.

        Example:
        -------
        models = {
            "linear": {
                "Ridge": RidgeCV(), 
                "Lasso": LassoCV(max_iter=5000, random_state=777), 
                "ElasticNet": ElasticNetCV(max_iter=5000, random_state=777)
            },
            "bagging": {
                "ExtraTrees": ExtraTreesRegressor(min_samples_leaf=5, max_features=0.33, 
                                                  n_estimators=100, random_state=777), 
                "RandomForest": RandomForestRegressor(min_samples_leaf=5, max_features=0.33, 
                                                      n_estimators=100, random_state=777)
            },
            "boosting": {
                "XGBoost": XGBRegressor(random_state=777), 
                "AdaBoost": AdaBoostRegressor(random_state=777)
            }
        }

    alphas : np.ndarray, optional
        Array of alpha values used for computing prediction intervals. Default is 1 - np.arange(0.01, 1.01, 0.01).

    n_boot : int, optional
        The number of bootstrapping iterations to perform. Default is 50.

    n_models : int, optional
        Number of top-performing models to select based on RMSE. Default is 3.

    """

    def __init__(self, models):
        # save the models: future interations will simplify this so we don't include model types
        self.models = models
        self.models_flat = {key: model for subdict in models.values() for key, model in subdict.items()}
        # flatten model dictionary
        flattened_dict = {}
        for subdict in models.values():
            for key, value in subdict.items():
                flattened_dict[key] = value
        self.models_flat  = flattened_dict

        self.trained = False
        self.calibrated = False
        self.predicted = False

    def train_predict(self, x_train, y_train, 
                      x_val, y_val, x_test, 
                      n_boot=100, alpha=0.1, init_seed=777,
                      save_option=None, file_name=None, load_option=None,
                      gamma_params=None, best="all",
                      n_models=3):
        if save_option is not None:
            _ = self.train(x_train = x_train, y_train = y_train,
                           x_val = x_val, y_val = y_val,
                           n_boot = n_boot, save_option = save_option,
                           file_name = file_name, init_seed = init_seed)
        else:
            _ = self.load_pre_train(file_name = file_name, 
                                    load_option = load_option, 
                                    n_boot = n_boot)
        
        __ = self.calibrate(x_val = x_val, y_val = y_val,
                            gamma_params = gamma_params, best = best,
                            alpha = alpha, n_models = n_models)
        
        pred_intervals = self.predict(x_test)
        return pred_intervals
    
    def train(self, x_train, y_train,
              x_val, y_val,
              n_boot=100,
              save_option=None, 
              file_name=None,
              init_seed = 777):

        """
        Fits models on training data, evaluates them on validation data, performs bootstrapping, 
        and saves the results to a specified directory.

        If you have already trained and saved the results, you can use the 
        `load_pre_train` method to load them instead of retraining.

        Returns:
        -------
        results_dict : dict
            Dictionary containing fitted models, validation results, and performance metrics.
        """
        # save the data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.n_boot = n_boot
        self.init_seed = init_seed

        # do pred-screening to filter out models
        model_performance_train = []
        model_performance = []

        for model_group, model_dict in self.models.items():
            for model_name, model in model_dict.items():
                fitted_model = model.fit(x_train, y_train)

                pred_train = fitted_model.predict(x_train)
                rmse_train = root_mean_squared_error(y_train, pred_train)
                mae_train = mean_absolute_error(y_train, pred_train)
                r2_train = r2_score(y_train, pred_train)
                model_performance_train.append([model_group,
                                                model_name,
                                                rmse_train,
                                                mae_train,
                                                r2_train])

                pred_val = fitted_model.predict(x_val)
                rmse_val = root_mean_squared_error(y_val, pred_val)
                mae_val = mean_absolute_error(y_val, pred_val)
                r2_val = r2_score(y_val, pred_val)
                model_performance.append([model_group, model_name, rmse_val, mae_val, r2_val])

        print("Evaluation complete, here's the result")

        performance_results_train = pd.DataFrame(model_performance_train, columns=["model_group", "model", "rmse", "mae", "r2"])
        performance_results = pd.DataFrame(model_performance, columns=["model_group", "model", "rmse", "mae", "r2"])

        print(performance_results)

        print("Bootstrapping")
        fitted_models = {model_group: {model_name: [] for model_name in model_dict}
                                for model_group, model_dict in self.models.items()}
        for i in range(n_boot):
            x_boot, y_boot = resample(x_train, y_train, random_state = self.init_seed + i)
            for model_group, model_dict in self.models.items():
                for model_name, model in model_dict.items():
                    boot_model = clone(model).fit(x_boot, y_boot)
                    fitted_models[model_group][model_name].append(boot_model)    

        results_dict = {
            "fitted_models": fitted_models, 
            "pred-screen-train": performance_results_train,
            "pred-screen": performance_results
        }

        if save_option is not None:
            if isinstance(save_option, str):
                if not os.path.exists(save_option):
                    os.makedirs(save_option)
                if file_name is None:
                    raise ValueError("Please provide a file name.")
                
                save_path = os.path.join(save_option, f"{file_name}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(results_dict, f)
                print(f"Results saved to {save_path}")
            elif callable(save_option):
                save_option(results_dict)
            else:
                raise ValueError("save_option must be a string or a callable function.")

        self.results_dict = results_dict
        self.trained = True
        
        return results_dict
    
    def load_pre_train(self, file_name=None, load_option=None, n_boot=100):
        """
        This method is useful for loading previously saved model results (e.g., fitted models,
        validation results, and performance metrics) without retraining. 
        
        If you have already called the `train` method, there is no need to use this function.
        """
        if isinstance(load_option, str):
            load_path = os.path.join(load_option, f"{file_name}.pkl")
        
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"No results file found at {load_path}")
            
            with open(load_path, "rb") as f:
                results_dict = pickle.load(f)
            
            print(f"Results loaded from {load_path}")
        elif callable(load_option):
            results_dict = load_option()
        self.n_boot = n_boot
        self.results_dict = results_dict
        self.trained = True

        return results_dict
    
    def calibrate(self, 
                  x_val = None, y_val = None,
                  gamma_params=None, best="all",
                  alpha=0.1, n_models=3):
        
        """
        Loads saved models and validation results, selects top-performing models,
        and calibrates gamma using the validation set.

        Parameters:
        -------
        gamma_params : dict, optional
            Dictionary of parameters to initialize gamma_classic. Example:
            {
                "selection_mode": "multiplicative",
                "fit_mode": "vanilla",
                "threshold": 0.1
            }
            These parameters are used to create a new instance of gamma_classic 
            to perform interval calibration based on empirical coverage of the validation data.

        best : str, optional
            Selection criterion for top-performing models, either 'all' or 'group'.

        Returns:
        -------
        alpha_results : list
            List of DataFrames of calibrated prediction intervals along with coverages and width.
        """
        if self.trained is False:
            raise ValueError("Please call 'train' first to fit the models.")
        
        self.gamma_algo = gamma_classic(
            selection_mode=gamma_params.get("selection_mode", "multiplicative"),
            fit_mode=gamma_params.get("fit_mode", "vanilla"),
            threshold=gamma_params.get("threshold", None),
            diagnosis=gamma_params.get("diagnosis", False),
            clip_mode = gamma_params.get("clip_mode", None),
            allow_shrinikng=gamma_params.get("allow_shrinikng", False)
        )
        self.alpha = alpha
        self.n_models = n_models
        
        print("processing data")

        results_dict = self.results_dict
        self.fitted_models = results_dict["fitted_models"]
        performance_results = results_dict["pred-screen"]


        self.x_val = x_val
        self.y_val = y_val
        val_results_dict = {}
        for model_group, model_dict in self.fitted_models.items():
            for model_name, boot_models in model_dict.items():
                model_results_val = []
                for i in range(self.n_boot):
                    model_results_val.append(boot_models[i].predict(x_val).tolist())
                val_results_dict[model_name] = pd.DataFrame(model_results_val)       

        # Select top models based on performance
        if best == "all":
            self.selected_models = performance_results.sort_values(by='rmse').head(self.n_models)['model']
        elif best == "group":
            self.selected_models = performance_results.groupby("model_group").apply(
                lambda x: x.nsmallest(1, "rmse")).reset_index(drop=True)["model"]
        else:
            raise Exception("'best' must be 'all' or 'group'")

        # Collect validation results for selected models
        self.val_results_df = pd.concat([val_results_dict[model] for model in self.selected_models], axis=0).transpose()

        # Calibrate gamma on the validation data
        self.gamma = self.gamma_algo.get_gamma(self.val_results_df, self.y_val, alpha=alpha)
        self.calibrated = True

        return self.gamma
        

    def predict(self, x_test):
        """
        Uses the calibrated gamma value to make predictions on the test set.
        
        Parameters:
        ----------
        x_test : pd.DataFrame
            Test data for prediction.
        val_results_df : pd.DataFrame
            Validation data used in gamma calibration.
        alpha : float, default=0.1
            Significance level for the intervals.
        threshold : float, optional
            Width threshold for the intervals.

        Returns:
        -------
        pd.DataFrame
            DataFrame of calibrated intervals for the test data.
        """

        # Check if the necessary attributes are defined
        if self.calibrated is False:
            raise ValueError("Please call 'calibrate' first to calibrate the gamma value.")

        # Load test results for selected models
        test_results = {model_name: [] for model_name in self.selected_models}
        for model_name in self.selected_models:
            model_group = [group for group in self.fitted_models if model_name in self.fitted_models[group]][0]
            for boot_model in self.fitted_models[model_group][model_name]:
                test_pred = boot_model.predict(x_test).tolist()
                test_results[model_name].append(test_pred)
        test_results_dict = {model_name: pd.DataFrame(test_results[model_name]) for model_name in test_results.keys()}        
        test_results_df = pd.concat([test_results_dict[model] for model in self.selected_models], axis=0).transpose()

        gamma_algo_copy = deepcopy(self.gamma_algo)
        test_ci_df = gamma_algo_copy.apply_gamma(test_results_df, alpha=self.alpha)
        test_ci_df["raw_width"] = gamma_algo_copy.raw_width
        test_ci_df["raw_coverage"] = gamma_algo_copy.raw_coverage
        self.prediction_intervals = test_ci_df
        self.test_results_df = test_results_df
        self.predicted = True

        return test_ci_df
    
    
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