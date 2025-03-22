#standard import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pyreadr
from copy import deepcopy

def determine_gammas(ci_df, y_val, selection_mode, return_error = False, allow_shrinking = False):
    gammas = []
    errors = []
    np_pointer = 0
    for i in ci_df.index.tolist():
        y = y_val[np_pointer]
        np_pointer += 1
        y_hat = ci_df.loc[i,"point_est"]
        lb = ci_df.loc[i, "lb"]
        ub = ci_df.loc[i, "ub"]
        error = abs(y - y_hat)
        errors.append(error)
        if selection_mode == "multiplicative":   
            if y >= y_hat:
                potential_gamma = (y - y_hat) / (ub - y_hat)
                if allow_shrinking:
                    gammas.append(potential_gamma)
                else:
                    gammas.append(max(potential_gamma, 1))
            else:
                potential_gamma = (y_hat - y) / (y_hat - lb)
                if allow_shrinking:
                    gammas.append(potential_gamma)
                else:
                    gammas.append(max(potential_gamma, 1))
        else:
            raise ValueError("Only multiplicative mode is supported")
    gammas_array = np.array(gammas)    
    if return_error:
        return gammas_array, errors
    else:
        return gammas_array
    
def get_interval_classic(row, gamma, alpha=0.1, selection_mode="multiplicative", 
                         clip = False, clip_threshold = None, clip_mode = None,
                         max_cal = None):
    med = np.nanquantile(row, 0.5)
    lb_orig = np.nanquantile(row, alpha/2)
    ub_orig = np.nanquantile(row, 1-alpha/2)
    
    if selection_mode == "multiplicative":
        if clip and ((ub_orig - lb_orig) > clip_threshold):
            if clip_mode == 'no_scale':
                lb = lb_orig
                ub = ub_orig
            elif clip_mode == 'perc_cal':
                gamma_star = clip_threshold / (ub_orig - lb_orig)
                lb = max(lb_orig, med - gamma_star * (med - lb_orig))
                ub = min(ub_orig, med + gamma_star * (ub_orig - med))
            elif clip_mode == 'max_cal':
                gamma_star = max_cal / (ub_orig - lb_orig)
                lb = max(lb_orig, med - gamma_star * (med - lb_orig))
                ub = min(ub_orig, med + gamma_star * (ub_orig - med))
            else:
                raise ValueError(f"Clip mode '{clip_mode}' is not supported")
        else:
            lb = med - gamma * (med - lb_orig)
            ub = med + gamma * (ub_orig - med)

    elif selection_mode == "additive":
        if clip and ((ub_orig - lb_orig) > clip_threshold):
            lb = lb_orig
            ub = ub_orig
        else:
            lb = lb_orig - gamma
            ub = ub_orig + gamma
        
    return [lb, ub, med]

class gamma_classic:
    def __init__(self, selection_mode = "multiplicative", diagnosis = False, 
                 fit_mode = "vanilla", threshold = None,
                 allow_shrinikng = False, clip_mode = None):
        # fit modes: vanilla, optbin
        self.selection_mode = selection_mode
        self.fit_mode = fit_mode
        self.diagnosis = diagnosis
        self.clip_mode = clip_mode
        self.threshold = threshold
        self.shrink = allow_shrinikng
        
        
    def get_gamma(self, val_results_df, y_val, alpha=0.1):
        """
        Calibrates the validation set and determine the gamma adjustment.

        Parameters:
        ----------
        alpha : float, default=0.1
            Coverage threshold

        Returns:
        -------
        float
            The gamma value for calibration.
        """

        self.alpha = alpha
        val_ci_df = val_results_df.apply(
            lambda r: [np.nanquantile(r, self.alpha / 2), np.nanquantile(r, 1 - self.alpha / 2), np.nanquantile(r, 0.5)],
            axis=1, result_type="expand"
        ).rename(columns={0: "lb", 1: "ub", 2: "point_est"})      
        # if self.threshold is not None:
        #     cutoff = np.nanquantile(val_ci_df["ub"] - val_ci_df["lb"], 1 - self.threshold)
        #     y_val = deepcopy(y_val[(val_ci_df["ub"] - val_ci_df["lb"]) <= cutoff])
        #     val_ci_df = val_ci_df[(val_ci_df["ub"] - val_ci_df["lb"]) <= cutoff].reset_index(drop=True)
        gammas = determine_gammas(ci_df=val_ci_df, y_val=y_val, selection_mode=self.selection_mode, 
                                  return_error=self.diagnosis, allow_shrinking = self.shrink)
        
        val_ci_df["gammas"] = gammas
        gamma_index = min(int(len(gammas) * self.alpha), len(gammas) - 1)
        self.gamma = val_ci_df.sort_values("gammas", ascending=False).iloc[gamma_index]["gammas"]

        self.raw_width = np.mean(val_ci_df["ub"] - val_ci_df["lb"])
        self.raw_coverage = np.mean((val_ci_df["lb"] <= y_val) & (val_ci_df["ub"] >= y_val))
        if self.threshold is not None:
            if self.clip_mode not in ['no_scale', 'perc_cal', 'max_cal']:
                raise ValueError(f"Clip mode '{self.clip_mode}' is not supported")
            self.cutoff = np.nanquantile(val_ci_df["ub"] - val_ci_df["lb"], 1 - self.threshold)
            if self.clip_mode == 'max_cal':
                self.max_width = np.max(val_ci_df["ub"] - val_ci_df["lb"])
            else:
                self.max_width = None
        if self.diagnosis:
            val_ci_df["errors"] = np.abs(y_val - val_ci_df["point_est"])


    def apply_gamma(self, test_results_df, alpha=0.1):
        """
        Applies the calibrated gamma value to the test data to calculate prediction intervals.

        Parameters:
        ----------
        test_results_df : pd.DataFrame
            DataFrame of test results.
        alpha : float, default=0.1
            Significance level for the intervals.
        threshold : float, optional
            Width threshold for the intervals.

        Returns:
        -------
        pd.DataFrame
            DataFrame of calibrated intervals for the test data.
        """
        if self.gamma is None:
            raise ValueError("Gamma has not been calibrated. Please run 'calibrate_validation' first.")
        
        self.alpha = alpha

        # No Threshold
        if self.threshold is None:
            if self.fit_mode == "vanilla":
                return test_results_df.apply(
                    lambda r: get_interval_classic(r, self.gamma, self.alpha, self.selection_mode, clip=False),
                    axis=1, result_type="expand"
                ).rename(columns={0: "lb", 1: "ub", 2: "point_est"})
            else:
                raise ValueError(f"Fit mode '{self.fit_mode}' is not supported for no threshold application")
        else:
            # With Threshold 
            if not (0 <= self.threshold <= 1):
                raise ValueError("Invalid threshold; must be between 0 and 1.")
            if self.fit_mode == "vanilla":
                width_threshold = self.cutoff
                return test_results_df.apply(
                    lambda r: get_interval_classic(r, self.gamma, self.alpha, self.selection_mode, 
                                                   clip=True, clip_threshold=width_threshold, 
                                                   clip_mode=self.clip_mode, max_cal=self.max_width),
                    axis=1, result_type="expand"
                ).rename(columns={0: "lb", 1: "ub", 2: "point_est"})
            else:
                raise ValueError(f"Fit mode '{self.fit_mode}' is not supported for threshold application")