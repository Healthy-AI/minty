from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from models.bst_util import *
from models.minty import MINTYRegression
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error


class Evaluation:
    def predict(self, X_test, I, estimator, classifier):
        # mse and CI
        if I is None:
            y_pred = estimator.predict(X_test).reshape(-1, 1)
            if classifier == True:
                y_pred_prob = estimator.predict_proba(X_test)
            else:
                y_pred_prob = None
        else:
            y_pred = estimator.predict(I.transform(X_test)).reshape(-1, 1)
            if classifier == True:
                y_pred_prob = estimator.predict_proba(X_test)
            else:
                y_pred_prob = None
        return y_pred, y_pred_prob

    def auc_ci(self, auc, n0, n1, alpha=0.05):
        q0 = auc / (2 - auc)
        q1 = 2 * auc * auc / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n0 - 1) * (q0 - auc * auc) + (n1 - 1) * (q1 - auc * auc)) / (n0 * n1))
        z = norm.ppf(1 - alpha / 2)
        ci = z * se

        return (auc - ci, auc + ci)

    def evaluate(self, y_pred, y_pred_prob, y_test, S, label, classifier):
        if classifier == True:
            # accuracy and CI
            acc = accuracy_score(y_test, y_pred)
            n = np.sum(y_test)
            interval = 1.96 * sqrt((acc * (1 - acc)) / n)
            acc_l = acc - interval
            acc_u = acc + interval

            # auc and CI
            auc = roc_auc_score(y_test, y_pred_prob[:, 1])

            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.values

            n_y_test_0 = np.sum(y_test == y_test.min())
            n_y_test_1 = np.sum(y_test == y_test.max())
            ci_0, ci_1 = Evaluation().auc_ci(auc, n_y_test_0, n_y_test_1, alpha=0.05)

            results = {'acc_' + label: [acc], 'acc_l_' + label: [acc_l], 'acc_u_' + label: [acc_u],
                       'auc_' + label: [auc], 'auc_l_' + label: [ci_0], 'auc_u_' + label: [ci_1]}
        else:
            #rescale y_test and y_pred
            y_test = S.inverse_transform(y_test)
            
            #If estimator == neumiss:
            y_pred = y_pred.reshape(-1, 1)
            y_pred = S.inverse_transform(y_pred)
            
            # mse and CI
            mse = mean_squared_error(y_test, y_pred)
            n = y_test.shape[0]
            interval = 1.96 * sqrt((2 * mse) / n)
            mse_l = mse - interval
            mse_u = mse + interval

            # r^2 and CI
            r_2 = r2_score(y_test, y_pred)

            # k = np.sum(X_test.columns)
            n = y_test.shape[0]
            interval = 1.96 * sqrt((1 - r_2) / (n - 2))
            r_2_l = r_2 - interval
            r_2_u = r_2 + interval
            
            #Root means square error 
            rmse = root_mean_squared_error(y_test, y_pred)
            
            
            results = {'mse_' + label: [mse], 'mse_l_' + label: [mse_l], 'mse_u_' + label: [mse_u],
                          'r_2_' + label: [r_2], 'r_2_l_' + label: [r_2_l], 'r_2_u_' + label: [r_2_u], 'rmse_' + label: [rmse]}

        return results

    def missingness_reliance(self, data, estimator, label):
        """
        Input: data: numpy array with float values for non-discretized data setting and int for discretized data setting
         Output: Find missingnes realiance which is the fraction of observations where the model relies either on an imputed value or,
         in the case of XGBoost, on a "default" child node.
         In other words, each observation imp_rel has reliance r_i which is either 0 or 1.
         For all methods, the average reliance is the average of r_i over a particular dataset.
        For methods using imputation,e.g. Neumiss, lasso, and Dt depend on all features, this is the same as the fraction of observations with
        at least one missing value in any feature.
        For XGBoost, it is the fraction of observations for which predictions relies on following at least one "default" node.
        for Rulefit, it is the fraction of observations for which at least one rule is unobserved or undetermined (0,0,nan) or (nan, nan) for example.
        For Minty, it is the fraction of observations for which at least one rule is unobserved or undetermined (0,0,nan) or (nan, nan) for example.
        This is the second number returned by the MINTYRegression.missingness_reliance() function.
        """
        data = data.astype(float)
        print("Heeeeellooo", data)
        #Transform estimator to string
        estimator_str = estimator.__class__.__name__.lower()

        if estimator_str == 'xgbregressor':
            imp_rel = bst_missingness_reliance(estimator, data)
        elif estimator_str == 'neumissmlp':
            imp_rel = np.isnan(data).any(axis=1).mean()
        elif estimator_str == 'lasso':
            imp_rel = (np.isnan(data) * (np.abs(estimator.coef_.reshape(1, -1)) > 0)).any(axis=1).mean()
        elif estimator_str == 'decisiontreeregressor':
            imp_rel = dt_missingness_reliance(estimator, data)
        elif estimator_str == 'rulefitregressor':
            imp_rel = rulefit_missingness_reliance(estimator, data)
        elif estimator_str == 'mintyregression':
            _, imp_rel = estimator.missingness_reliance(data)
        else:
            imp_rel = None
            print("No missingness reliance defined for this estimator")

        missingness_R = {'missingness_reliance_' + label: [imp_rel]}

        return missingness_R





