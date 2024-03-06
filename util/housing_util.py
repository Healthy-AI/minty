import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from util.missingness import Missingness
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class preprocessing_housing:
    '''
    X_df = X [n, d], raw data
    X = X [n, d], preprocessed in bins 
    Xm = X with missing values [n, d] 
    y = y [n, 1] Outcome variable normalized
    '''

    def __init__(self, m=None):
        self.m = m


    def get_data(self):
        house_data = pd.read_csv(r'../data/datasets/housing.csv')
        house_data = pd.DataFrame(house_data)

        #removed

        house_features= ['MSZoning', 'LotArea',
         'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st',
                'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', 'GrLivArea', 'FullBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType',
           'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive',  'OpenPorchSF',
            'MoSold', 'YrSold', 'SaleType',
           'SaleCondition']

        X_df = house_data[house_features]

        cols = X_df.columns

        #Outcome variable Sales price
        outcome = house_data['SalePrice']
        Y = outcome.values.reshape(-1, 1) # this is the max 755000 this is the min 34900 this is the avg 180921.19589041095
        
       #numerical features that are not split by median
        num_cols_all = X_df._get_numeric_data().columns
        # remove ones that are only split by median
        num_special = ['OverallCond', 'FullBath', 'OpenPorchSF', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces',
                       'GarageCars']
        X_df[num_special] = X_df[num_special].apply(pd.to_numeric)

        num_cols = list(set(num_cols_all) - set(num_special))

        # categorical features
        cat_cols = list(set(cols) - set(num_cols)- set(num_special))

        # add missingness MCAR
        X_miss, M = Missingness().get_missingness(X_df, p=0.1)

        # add missingness MNAR or MAR
        #X_miss, M = Missingness().add_missingness(X_df, cat_cols, num_cols_all)


        #add missingness MCAR custom - correlated numeric features
        #X_miss_num, M_num = Missingness().find_correlated_pairs_num(X_df[num_cols_all], 0.6)
        #X_miss_cat, M_cat = Missingness().find_correlated_pairs_cat(X_df[cat_cols], 0.6)
        #X_miss = pd.concat((X_miss_cat, X_miss_num), axis=1)

        return X_miss, Y, cat_cols, num_cols, num_special

    def form_bins_num(self, X_miss, num_cols):
        num_X = X_miss[num_cols]
        X_bins = []
        bin_edges = []
        # fit by column not at once
        for column in num_X:
            # M_nan is a boolean matrix, True if the value is nan
            # Without values it is only an array
            M_nan = np.isnan(num_X[column].values)
            df_drop = num_X[column].dropna().values
            df_fill = num_X[column].fillna(0).values
            # ‘quantile’: All bins in each feature have the same number of points.
            binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile').fit(df_drop.reshape(-1, 1))
            # Transform each column
            bins_ = binner.transform(df_fill.reshape(-1, 1))
            bins_[M_nan, :] = np.nan  # set nan values back to nan
            X_bins.append(bins_)
            bin_edge_arr = binner.bin_edges_[0]
            # With strategy "quantiles", Some arrays only have 4 values, because of small sample sizes
            bin_edges.append(bin_edge_arr[1:len(bin_edge_arr) - 1])  # cut the first and the last value in each array
            # bin_edges.append(bin_edge_arr)

        a = np.array(X_bins)
        a = a.transpose(1, 0, 2).reshape(-1, a.shape[0])

        # Create Dataframe with transformed bin values
        X_bins_ = pd.DataFrame(a, index=num_X.index, columns=num_X.columns)

        return X_bins_, bin_edges


    def create_columnnames_num(self, bin_edges, num_X):
        new_columns_names = []
        origi_column_names = num_X.columns

        # Some arrays in vv don't have 3 values -> str(vv[counter][2] doesn't work!
        counter = 0
        for d in origi_column_names:
            columns = (str(d)) + " <= " + str(bin_edges[counter][0])
            new_columns_names.append(columns)
            columns = (str(d)) + " " + (str(bin_edges[counter][0]) + " - " + str(bin_edges[counter][1]))
            new_columns_names.append(columns)
            columns = (str(d)) + " " + (str(bin_edges[counter][1]) + " - " + str(bin_edges[counter][2]))
            new_columns_names.append(columns)
            columns = (str(d)) + " >= " + str(bin_edges[counter][2])

            new_columns_names.append(columns)
            counter += 1

        return new_columns_names

    def create_df_num(self, X_bins_, new_columns_names):
        ar = np.zeros((len(X_bins_), len(new_columns_names)))

        counter = 0  # goes over patients

        for ind in X_bins_:
            counter1 = 0
            for value in X_bins_[ind]:
                if value == 0.0:
                    ar[counter1, counter * 4] = 1
                elif value == 1.0:
                    ar[counter1, counter * 4 + 1] = 1
                elif value == 2.0:
                    ar[counter1, counter * 4 + 2] = 1
                elif value == 3.0:
                    ar[counter1, counter * 4 + 3] = 1
                else:
                    ar[counter1, counter * 4] = np.nan
                    ar[counter1, counter * 4 + 1] = np.nan
                    ar[counter1, counter * 4 + 2] = np.nan
                    ar[counter1, counter * 4 + 3] = np.nan
                counter1 += 1
            counter += 1

        df_num_bins = pd.DataFrame(ar, index=np.arange(len(X_bins_)), columns=new_columns_names)

        return df_num_bins

    def discretize_num(self, X_miss, num_cols):
        num_X = X_miss[num_cols]
        X_bins_, bin_edges = self.form_bins_num(X_miss, num_cols)
        new_columns_names = self.create_columnnames_num(bin_edges, num_X)
        df_num_bins = self.create_df_num(X_bins_, new_columns_names)

        return df_num_bins

    def binarize_special(self, X_miss, spec_col):
        X_spec = X_miss[spec_col].copy()
        X_special_ = pd.DataFrame()
        X_special_["OverallCond_split"] = (X_spec.OverallCond > X_spec.OverallCond.median()).replace(
            {True: 1, False: 0})
        X_special_["FullBath_split"] = (X_spec.FullBath > X_spec.FullBath.median()).replace({True: 1, False: 0})
        X_special_["OpenPorchSF_split"] = (X_spec.OpenPorchSF > X_spec.OpenPorchSF.median()).replace(
            {True: 1, False: 0})
        X_special_["BedroomAbvGr_split"] = (X_spec.BedroomAbvGr > X_spec.BedroomAbvGr.median()).replace(
            {True: 1, False: 0})
        X_special_["KitchenAbvGr_split"] = (X_spec.KitchenAbvGr > X_spec.KitchenAbvGr.median()).replace(
            {True: 1, False: 0})
        X_special_["Fireplaces_split"] = (X_spec.Fireplaces > X_spec.Fireplaces.median()).replace(
            {True: 1, False: 0})
        X_special_["GarageCars_split"] = (X_spec.GarageCars > X_spec.GarageCars.median()).replace(
            {True: 1, False: 0})

        # Create an instance of One-hot-encoder
        enc = OneHotEncoder()
        # Passing encoded columns
        df_num_bins_special = pd.DataFrame(enc.fit_transform(
            X_special_).toarray())
        df_num_bins_special.columns = ['OverallCond > 5', 'OverallCond < 5', 'FullBath > 2', 'FullBath < 2',
                                       'OpenPorchSF > 25', 'OpenPorchSF  < 25',
                                       'BedroomAbvGr > 3', 'BedroomAbvGr < 3', 'KitchenAbvGr > 1', 'KitchenAbvGr < 1',
                                       'Fireplaces < 1', 'Fireplaces > 1',
                                       'GarageCars > 2', 'GarageCars < 2']

        return df_num_bins_special


    def get_cat_dummies(self, X_miss, cat_cols):
        X_cat = X_miss[cat_cols].copy()
        cat_dummies_ = pd.get_dummies(X_cat, drop_first=True, dummy_na=True)
        cat_dummies = cat_dummies_
        #cat_dummies = cat_dummies_.astype(int)

        for c in cat_cols:
            cc = c + '_nan'
            # Replace all 1s with np.nan
            cat_dummies.loc[cat_dummies[cc] == 1, cat_dummies.columns != cc] = pd.NA

        df_cat_dummies = cat_dummies.replace({pd.NA: np.nan})
        # List comprehension to identify columns to remove
        columns_to_remove = [col for col in df_cat_dummies.columns if '_nan' in col]
        # Remove the identified columns
        df_cat_dummies.drop(columns=columns_to_remove, inplace=True)

        return df_cat_dummies

    def create_binarized_df(self, X_miss, cat_cols, num_cols, num_special):
        df_cat_dummies = self.get_cat_dummies(X_miss, cat_cols)
        df_num_bins = self.discretize_num(X_miss, num_cols)
        df_num_bins_special = self.binarize_special(X_miss, num_special)
        df_cat_dummies.reset_index(drop=True, inplace=True)
        df_num_bins.reset_index(drop=True, inplace=True)
        df_num_bins_special.reset_index(drop=True, inplace=True)
        #combine all dataframes to one
        X_m = pd.concat([df_num_bins, df_num_bins_special, df_cat_dummies], axis=1)

        return X_m