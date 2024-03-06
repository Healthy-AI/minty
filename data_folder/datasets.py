import numpy as np
import pandas as pd
import os
from data.synth_data import *
from util.adni_util import *
from util.housing_util import *
from util.life_util import *

class Create_data:
    """
        Load SYNTH data set
        Load ADNI data set
        Load Housing data set
        Load Life data set
        """

    def get_synth(self):
        X_m, Y, M = generate_data(n=50, c=15, delta=0.1, gamma=0.2) #new missingness generation process
        columns = ['X' + str(i) for i in range(1, X_m.shape[1] + 1)]
        X_df = pd.DataFrame(X_m, columns=columns)
        return X_m, Y, X_df

    def get_ADNI(self):
        df = pd.read_csv(r'../data/datasets/ADNI_non_discretized.csv')
        #df = pd.read_csv(r'../data/datasets/ADNI_discretized.csv')
        X_m = df.loc[:, df.columns != 'ADAS13']
        Y = df['ADAS13'].values.reshape(-1, 1)
        X_m = X_m.to_numpy()
        #X_m = X_m.astype(float)
        X_df = pd.DataFrame(X_m, columns=df.columns[:-1])
        return X_m, Y, X_df

    def get_housing(self):
        X_miss, Y, cat_cols, num_cols, num_special = preprocessing_housing().get_data()
        df_cat_dummies = preprocessing_housing().get_cat_dummies(X_miss, cat_cols) #encode categoricals in missingness frame
        X_miss = X_miss.drop(columns=cat_cols).copy() #remove orginial cat_columns form df with missingness
        X_m = pd.concat([X_miss, df_cat_dummies], axis= 1) #concatenate encoded categoricals to df with missingness
        #X_m = preprocessing_housing().create_binarized_df(X_miss, cat_cols, num_cols, num_special)
        X_df = pd.DataFrame(X_m, columns=X_m.columns)
        X_m = X_m.to_numpy()
        X_m = X_m.astype(float)
        
        return X_m, Y, X_df

    def get_life(self):
        X_miss, Y, spec_cols, num_cols, cat_cols = preprocessing_life().get_data()
        df_cat_dummies = preprocessing_housing().get_cat_dummies(X_miss, cat_cols)  # encode categoricals in missingness fram
        X_miss = X_miss.drop(columns=cat_cols).copy()  # remove orginial cat_columns form df with missingness
        X_m = pd.concat([X_miss, df_cat_dummies], axis=1)  # concatenate encoded categoricals to df with missingness
        #X_m = preprocessing_life().create_binarized_df(X_miss, cat_cols, num_cols, spec_cols)
        X_df = pd.DataFrame(X_m, columns=X_m.columns)
        X_m = X_m.to_numpy()
        return X_m, Y, X_df

