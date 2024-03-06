import numpy as np
import argparse
import torch
from data.datasets import *

from sklearn.model_selection import train_test_split
from util.preprocessing import *
from util.adni_util import *
from util.housing_util import *
from util.life_util import *
from sklearn.preprocessing import StandardScaler

class Data_Loader:
    def __init__(self, seed=None, dataset=None, imputation=None, split=None, estimator=None):
        self.seed = seed
        self.dataset = dataset
        self.imputation = imputation
        self.split = split
        self.estimator = estimator

    def load_data(self):
        """Loads the data sets

        Returns:
            Dataframes with Training, validation and test data
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        

        Crt_data = Create_data()
        
        if self.dataset == 'synth':
            X_m, Y, X_df = Crt_data.get_synth()
            X_train, X_test, y_train, y_test = train_test_split(X_m, Y, test_size=self.split)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.split)
            I = Preprocessing(X_train=X_train, seed=self.seed, imputation=self.imputation).imputer()
            S = StandardScaler().fit(y_train)
            y_train = S.transform(y_train)
            y_val = S.transform(y_val)
            y_test = S.transform(y_test)
        elif self.dataset == 'ADNI':
            X_m, Y, X_df = Crt_data.get_ADNI(estimator= self.estimator)
            X_train, X_test, y_train, y_test = train_test_split(X_m, Y, test_size=self.split)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.split)
            I = Preprocessing(X_train=X_train, seed=self.seed, imputation=self.imputation).imputer()
            S = StandardScaler().fit(y_train)
            y_train = S.transform(y_train)
            y_val = S.transform(y_val)
            y_test = S.transform(y_test)
        elif self.dataset == 'housing':
            X_m, Y, X_df = Crt_data.get_housing(estimator= self.estimator)
            X_train, X_test, y_train, y_test = train_test_split(X_m, Y, test_size=self.split)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.split)
            I = Preprocessing(X_train=X_train, seed=self.seed, imputation=self.imputation).imputer()
            S = StandardScaler().fit(y_train)
            y_train = S.transform(y_train)
            y_val = S.transform(y_val)
            y_test = S.transform(y_test)
        elif self.dataset == 'life':
            X_m, Y, X_df = Crt_data.get_life(estimator= self.estimator)
            X_train, X_test, y_train, y_test = train_test_split(X_m, Y, test_size=self.split)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.split)
            I = Preprocessing(X_train=X_train, seed=self.seed, imputation=self.imputation).imputer()
            S = StandardScaler().fit(y_train)
            y_train = S.transform(y_train)
            y_val = S.transform(y_val)
            y_test = S.transform(y_test)
        else:
            raise Exception('Unrecognized dataset: %s' % self.dataset)

        return I,S, X_train, X_test, X_val, y_train, y_test, y_val, X_df
