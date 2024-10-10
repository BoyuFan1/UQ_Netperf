class Conformal_UQ_roc:
    def __init__(self, model, alphas=np.arange(0.01, 1.01, 0.01)):
        self.model = model
        self.alphas = alphas
        
    def fit_predict(self, x_test, x_train_val = "None", y_train_val = "None", x_train="None", y_train="None", x_val="None", y_val="None"):
        """
        Conformal interval method.
        """
        if type(x_train) == str:
            split_index = int(len(x_train) * 0.8)
            x_val, y_val = x_train_val[split_index:], y_train_val[split_index:]
            x_train, y_train = x_train_val[:split_index], y_train_val[:split_index]

        m = self.model.fit(x_train, y_train)

        y_calib_pred = m.predict(x_val)
        calib_errors = np.abs(y_val - y_calib_pred)
        y_pred_val = m.predict(x_test)
        
        alpha_results = []
        for alpha in self.alphas:
            q = np.quantile(calib_errors, 1 - alpha)
            print(q)

            lower_bounds = y_pred_val - q
            upper_bounds = y_pred_val + q

            alpha_results.append(pd.DataFrame({"lb": lower_bounds, "ub": upper_bounds, "point_est": y_pred_val}))
            
        return alpha_results
