from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self, X_train=None, seed=None, imputation=None):
        self.X_train = X_train
        self.seed = seed
        self.imputation = imputation

    def imputer(self):
        if self.imputation == "zero":
            I = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0).fit(self.X_train)
        elif self.imputation == 'mice':
            I = IterativeImputer(max_iter=10, initial_strategy='mean', sample_posterior=True).fit(self.X_train)
        elif self.imputation == 'single':
            I = SimpleImputer(missing_values=np.nan, strategy="mean").fit(self.X_train)
        elif self.imputation == 'none':
            I = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=np.nan).fit(self.X_train)

        else:
            raise Exception('Unknown Imputation method; %s' % (self.imputation))

        return I
