import sys
sys.path.append('../')
import numpy as np
from models import minty
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from models.Neumiss.neumiss import NeuMissMLP
import imodels
from models.imodels.imodels import RuleFitRegressor 



dataset = ['synth', 'ADNI', 'housing', 'life']

N_SEEDS = 10
SEEDS = range(N_SEEDS)
N_ITERATIONS = 5
ITERATIONS = range(N_ITERATIONS)

def get_estimators():
    #Dictionaries including all estimators
    estimators = {
        'minty': minty.MINTYRegression,
        'lasso': linear_model.Lasso,
        'xgbr': XGBRegressor,
        'dt': DecisionTreeRegressor,
        'neumiss': NeuMissMLP,
        'rulefit': RuleFitRegressor,
    }
    return estimators

def config_synth():

    estimators = [get_estimators()[e] for e in ['minty', 'lasso', 'xgbr', 'dt', 'neumiss', 'rulefit']]
    paramtrs_all_e = {
        'minty': {'optimizer': ['ilp'], 'classifier': [False], 'max_rules': [10], 'lambda_0': [1e-3, 0.01],
                  'lambda_1': [1e-3, 0.01], 'gamma': [0, 1e-3, 0.01, 10000],
                  'reg_refit': [True], 'relaxed': [True], 'optimality_tol': [1e-6], 'feasibility_tol': [1e-6],
                  'reg_rho': [False, True], 'silent': [False], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'minty': {'optimizer': ['beam'], 'classifier': [False], 'max_rules': [10], 'lambda_0': [1e-3, 0.01, 0.1], 'lambda_1': [0.01, 0.1],
                  'gamma': [0, 1e-3, 0.01, 0.1, 10000],
                  'reg_refit': [True], 'relaxed': [True], 'optimality_tol': [1e-6], 'feasibility_tol': [1e-6],
                 'beam_width': [77], 'beam_depth': [7], 'reg_rho': [False, True], 'silent': [False], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'lasso': {'alpha': [0.01, 0.1, 0.2, 0.4], 'fit_intercept': [True], 'precompute': [True], 'i': ['zero', 'mice'], 'sp': [0.2],
                  's': SEEDS, 'it': [N_ITERATIONS]},
        'xgbr': {'learning_rate': [0.2, 0.3], 'max_depth': [4], 'lambda': [0.5], 'alpha': [0.2], 'i': ['none'], 'sp': [0.2],
                 's': SEEDS, 'it': [1]},
        'dt': {'criterion': ['squared_error'], 'splitter': ['best'], 'min_samples_leaf': [10, 20, 50], 'min_impurity_decrease': [0.1],
               'ccp_alpha': [0, 0.005], 'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'neumiss': {'n_features': [30], 'neumiss_depth': [3, 5], 'mlp_depth': [7, 10], 'mlp_width': [30], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'rulefit': {'max_rules': [7, 10, 15], 'sample_frac': [0.1, 0.2], 'tree_size':[5, 10, 15], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]}
    }
    return estimators, paramtrs_all_e


def config_ADNI():

    estimators = [get_estimators()[e] for e in ['minty', 'lasso', 'xgbr', 'dt', 'neumiss', 'rulefit']]

    paramtrs_all_e = {
        'minty': {'optimizer': ['beam'], 'classifier': [False], 'max_rules': [10], 'lambda_0': [1e-3, 0.01, 0.1],
                  'lambda_1': [1e-3, 0.01, 0.1], 'gamma': [0, 1e-7, 1e-3, 0.01, 0.1, 10000],
                  'reg_refit': [True], 'relaxed': [True], 'optimality_tol': [1e-6], 'feasibility_tol': [1e-6],
                  'beam_width': [77], 'beam_depth': [7], 'reg_rho': [False, True], 'silent': [False], 'i': ['none'],
                  'sp': [0.2], 's': SEEDS, 'it': [1]},
        'lasso': {'alpha': [0.01, 0.1, 0.2, 0.4], 'fit_intercept': [True], 'precompute': [True],
                  'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'it': [N_ITERATIONS]},
        'xgbr': {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'max_depth': [4, 6, 10], 'lambda': [0.5, 1], 'alpha': [0, 0.1, 0.2, 0.3],
                'i': ['none'], 'sp': [0.2],'s': SEEDS,  'it': [1]},
        'dt': {'criterion': ['squared_error'], 'splitter': ['best'], 'min_samples_leaf': [10, 20, 50],  'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'neumiss': {'n_features': [77], 'neumiss_depth': [3, 5], 'mlp_depth': [7, 10], 'mlp_width': [77], 'i': ['none'],
                    'sp': [0.2],'s': SEEDS, 'it': [1]},
        'rulefit': {'max_rules': [7, 10, 15], 'sample_frac': [0.1, 0.2], 'tree_size':[5, 10, 15],'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]}
    }
    return estimators, paramtrs_all_e


def config_housing():
    estimators = [get_estimators()[e] for e in ['minty', 'lasso', 'xgbr', 'dt', 'neumiss', 'rulefit']]

    paramtrs_all_e = {
        'minty': {'optimizer': ['beam'], 'classifier': [False], 'max_rules': [20], 'lambda_0': [1e-3, 0.01, 0.1],
                  'lambda_1': [1e-3, 0.01, 0.1], 'gamma': [0, 1e-7, 1e-3, 0.01, 0.1, 10000],
                  'reg_refit': [True], 'relaxed': [True], 'optimality_tol': [1e-6], 'feasibility_tol': [1e-6],
                  'beam_width': [77], 'beam_depth': [7], 'reg_rho': [False, True], 'silent': [False],'i': ['none'],
                  'sp': [0.2], 's': SEEDS, 'it': [1]},
        'lasso': {'alpha': [0.01, 0.1, 0.2, 0.4], 'fit_intercept': [True], 'precompute': [True],
                 'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'it': [N_ITERATIONS]},
        'xgbr': {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'max_depth': [4, 6, 10], 'lambda': [0.5, 1], 'alpha': [0, 0.1, 0.2, 0.3],
                 'i': ['none'], 'sp': [0.2],'s': SEEDS, 'it': [1]},
        'dt': {'criterion': ['squared_error'], 'splitter': ['best'], 'min_samples_leaf': [10, 20, 50], 'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'it': [1]},
        'neumiss_1': {'n_features': [217], 'neumiss_depth': [3, 5], 'mlp_depth': [7, 10], 'mlp_width': [217], 'i': ['none'], 'sp': [0.2], 's': [1], 'c': [0], 'it': [1]},
        'rulefit': {'max_rules': [7, 10, 15], 'sample_frac': [0.1, 0.2], 'tree_size':[5, 10, 15], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]}
    }
    return estimators, paramtrs_all_e

def config_life():
    estimators = [get_estimators()[e] for e in ['minty', 'lasso', 'xgbr', 'dt', 'neumiss', 'rulefit']]
    
    paramtrs_all_e = {
        'minty': {'optimizer': ['beam'], 'classifier': [False], 'max_rules': [20], 'lambda_0': [1e-3, 0.01, 0.1],
                  'lambda_1': [1e-3, 0.01, 0.1], 'gamma': [0, 1e-7, 1e-3, 0.01, 0.3, 0.5, 10000], 'reg_refit': [True], 'relaxed': [True], 'optimality_tol': [1e-6], 'feasibility_tol': [1e-6],
                 'beam_width': [77], 'beam_depth': [7], 'reg_rho': [False, True], 'silent': [False], 'i': ['none'],'sp': [0.2], 's': SEEDS, 'c': [0], 'it': [1]},
        'lasso': {'alpha': [0.01, 0.1, 0.2, 0.4], 'fit_intercept': [True], 'precompute': [True],
                'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'c': [0],  'it': [N_ITERATIONS]},
        'xgbr': {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'max_depth': [4, 6, 10], 'lambda': [0.5, 1], 'alpha': [0, 0.1, 0.2, 0.3],
                 'i': ['none'], 'sp': [0.2],'s': SEEDS, 'c': [0], 'it': [1]},
        'dt': {'criterion': ['squared_error'], 'splitter': ['best'], 'min_samples_leaf': [10, 20, 50],  'i': ['zero', 'mice'], 'sp': [0.2], 's': SEEDS, 'c': [0], 'it': [1]},
        'neumiss': {'n_features': [64], 'neumiss_depth': [3, 5], 'mlp_depth': [7, 10], 'mlp_width': [64], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'c': [0], 'it': [1]},
        'rulefit': {'max_rules': [7, 10, 15], 'tree_size':[5, 10, 15], 'i': ['none'], 'sp': [0.2], 's': SEEDS, 'it': [1]}
    }
    return estimators, paramtrs_all_e
