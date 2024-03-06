import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer


def load_ADNIRegressor(base_path='', frac=1.0):
    # read csv files and create dataframe
    df = pd.read_csv(os.path.join(base_path, '../data_folder/ADNIMERGE.csv'))
    df = pd.DataFrame(df)

    # Filter VISCODE columns
    df = df[(df['VISCODE'] == 'bl') | (df['VISCODE'] == 'm24')]
    # sort RID values for bl and m24
    df_1 = df.sort_values(by=['RID'])

    # Filter for rows that have entry in bl and a unique RID
    bl_ids = df_1[(df_1['VISCODE'] == 'bl') & (df_1['DX'].notna())]['RID'].unique()

    # Filter the baseline corhort for only entries that have a follow up entry at m24
    m24_ids = df_1[(df_1['VISCODE'] == 'm24') & (df_1['DX'].notna())]['RID'].unique()

    # Find overlap between dfs
    li = np.intersect1d(bl_ids, m24_ids)
    df_2 = df_1[df_1['RID'].isin(li)]

    # sort by RID
    loaded_data = df_2.sort_values(by=['RID'])

    # Change all < to only a integer
    loaded_data.loc[:, 'TAU_bl'] = loaded_data['TAU_bl'].replace('<80', '80').astype(np.float64)
    loaded_data.loc[:, 'ABETA_bl'] = loaded_data['ABETA_bl'].replace('>1700', '1700').replace('<200', '200').astype(
        np.float64)
    loaded_data.loc[:, 'PTAU_bl'] = loaded_data['PTAU_bl'].replace('<8', '8').astype(np.float64)

    # filter bl otherwise m24 is also in the training
    data = loaded_data.loc[(loaded_data['VISCODE'] == 'bl')].sort_values(by=['RID'])
    data_y = loaded_data.loc[(loaded_data['VISCODE'] == 'm24')].sort_values(by=['RID'])

    # Only select rows that have a value in ADAS13
    data = data[data['ADAS13'].notna()]

    data = data[['AGE', 'PTEDUCAT', 'ADAS13', 'APOE4', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'LDELTOTAL', 'MMSE',
                 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV', 'DX_bl', 'PTGENDER',
                 'PTETHCAT',
                 'PTRACCAT', 'PTMARRY']].copy()

    if frac < 1.0:
        data = data.sample(frac=frac)

    # select all columns but DX = Target variable
    X = data.loc[:, data.columns != 'ADAS13']  # select all columns but ADAS13
    Y_ = data.loc[:, data.columns == 'ADAS13']  # select ADAS13 = Target variable

    # Standardize Y
    Y = StandardScaler().fit_transform(Y_)

    return X, Y


c_cont_ADNI = ['AGE', 'PTEDUCAT', 'APOE4', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'LDELTOTAL', 'MMSE',
               'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV']


class Standardizer(StandardScaler):
    """
    Standardizes a subset of columns using the scikit-learn StandardScaler
    """

    def __init__(self, copy=True, with_mean=True, with_std=True, columns=None, ignore_missing=True):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.ignore_missing = ignore_missing

    def fit(self, X, y=None):
        columns = X.columns if self.columns is None else self.columns

        StandardScaler.fit(self, X[columns], y)

        return self

    def transform(self, X, copy=None):
        columns = X.columns if self.columns is None else self.columns

        Xn = X.copy()
        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            columns_mis = [c for c in columns if c not in X.columns]

            if len(columns_sub) == 0:
                return X

            Xt = X.copy()
            Xt[columns_mis] = 0
            try:
                Xt = StandardScaler.transform(self, Xt[columns_sub + columns_mis], copy=copy)
            except:
                print(columns_sub + columns_mis)
                print(Xt[columns_sub + columns_mis])

            Xt = Xt[:, :len(columns_sub)]
            Xn.loc[:, columns_sub] = Xt
        else:
            print('here are columns', columns)
            Xt = StandardScaler.transform(self, X[columns], copy=copy)
            Xn.loc[:, columns] = Xt

        return Xn

    def inverse_transform(self, X, copy=None):
        columns = X.columns if self.columns is None else self.columns

        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            Xn = self.inverse_transform_single(X, columns_sub, copy=copy)
        else:
            Xt = StandardScaler.inverse_transform(self, X[columns], copy=copy)
            Xn = X.copy()
            Xn.loc[:, columns] = Xt

        return Xn

    def inverse_transform_single(self, Xs, columns, copy=None):
        X = pd.DataFrame(np.zeros((Xs.shape[0], len(self.columns))), columns=self.columns)
        X[columns] = Xs[columns]

        Xt = StandardScaler.inverse_transform(self, X[self.columns], copy=copy)
        X.loc[:, self.columns] = Xt

        return X[columns]

    def fit_transform(self, X, y=None, **fit_params):
        Xt = StandardScaler.fit_transform(self, X, y, **fit_params)
        return Xt


class OneHotEncoderMissing(_BaseEncoder):

    def __init__(self, *,
                 keep_nan=True,
                 return_df=True,
                 sparse=False,
                 categories="auto",
                 drop=None,
                 dtype=np.float64,
                 handle_unknown="error"):

        self.ohe = OneHotEncoder(sparse=sparse, categories=categories, drop=drop,
                                 dtype=dtype, handle_unknown=handle_unknown)

        if sparse:
            raise Exception('Sparse mode not supported.')

        self.keep_nan = keep_nan
        self.return_df = return_df
        self.sparse = sparse
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.ohe.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.ohe.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        if self.keep_nan:
            return [c for c in self.ohe.get_feature_names_out(input_features) if not c[-4:] == '_nan']
        else:
            return self.ohe.get_feature_names_out(input_features)

    def inverse_transform(self, X):
        cols = self.ohe.feature_names_in_

        if self.keep_nan:
            X_t = X.copy()
            if not isinstance(X_t, pd.DataFrame):
                X_t = pd.DataFrame(X_t, columns=self.get_feature_names_out())

            oh_cols = self.ohe.get_feature_names_out()

            for c in cols:
                c_nan = '%s_nan' % c
                if c_nan in oh_cols:
                    cs_notnan = [ohc for ohc in oh_cols if (c + '_') in ohc and '_nan' not in c]
                    X_t[c_nan] = 1 * (X_t[cs_notnan[0]].isna())
                    X_t[cs_notnan] = X_t[cs_notnan].fillna(0)

            X = X_t[oh_cols]

        X_i = self.ohe.inverse_transform(X)
        if self.return_df:
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(X_i, columns=cols, index=X.index)
            else:
                return pd.DataFrame(X_i, columns=cols)
            return
        else:
            return X_i

    def transform(self, X):
        X_t = self.ohe.transform(X)

        oh_cols = self.ohe.get_feature_names_out()

        if not self.keep_nan:
            if self.return_df:
                if isinstance(X, pd.DataFrame):
                    return pd.DataFrame(X_t, columns=oh_cols, index=X.index)
                else:
                    return pd.DataFrame(X_t, columns=oh_cols)
            else:
                return X_t

        if isinstance(X, pd.DataFrame):
            df_t = pd.DataFrame(X_t, columns=oh_cols, index=X.index)
        else:
            df_t = pd.DataFrame(X_t, columns=oh_cols)

        cols = self.ohe.feature_names_in_

        for c in cols:
            c_nan = '%s_nan' % c
            if c_nan in oh_cols:
                cs_nan = [ohc for ohc in oh_cols if (c + '_') in ohc]
                df_t.loc[df_t[c_nan] > 0, cs_nan] = np.nan
                df_t.drop(columns=[c_nan], inplace=True)

        if self.return_df:
            return df_t
        else:
            return df_t.values


class Binning:
    # Bins can only be formed on observed data - not on nan
    def form_bins(self, X_num):
        df = X_num
        X_bins = []
        bin_edges = []
        # fit by column not at once
        for column in df:
            # M_nan is a boolean matrix, True if the value is nan
            #Without values it is only an array
            M_nan = np.isnan(df[column].values)
            df_drop = df[column].dropna().values
            df_fill = df[column].fillna(0).values
            # Note for starteg parameter: ‘uniform’: All bins in each feature have identical widths.
            # ‘quantile’: All bins in each feature have the same number of points.
            binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile').fit(df_drop.reshape(-1, 1))
            # Transform each column
            bins_ = binner.transform(df_fill.reshape(-1, 1))
            bins_[M_nan, :] = np.nan # set nan values back to nan
            #print(bins_[M_nan, :])
            X_bins.append(bins_)
            bin_edge_arr = binner.bin_edges_[0]
            # With strategy "quantiles", Some arrays only have 4 values, because of small sample sizes
            bin_edges.append(bin_edge_arr[1:len(bin_edge_arr) - 1])  # cut the first and the last value in each array

        a = np.array(X_bins)
        a = a.transpose(1, 0, 2).reshape(-1, a.shape[0])

        # Create Dataframe with transformed bin values
        X_bins_ = pd.DataFrame(a, index=X_num.index, columns=X_num.columns)

        return X_bins_, bin_edges

    def create_quantile_df_num(self, bin_edges, X_bins_):
        vv = bin_edges
        c = []
        column_names = X_bins_.columns

        # Some arrays in vv don't have 3 values -> str(vv[counter][2] doesn't work!
        counter = 0
        for d in column_names:
            columns = (str(d)) + " less than or equal " + str(vv[counter][0])
            c.append(columns)
            columns = str(vv[counter][0]) + " less than " + (str(d)) + " less than or equal " + str(vv[counter][1])
            c.append(columns)
            columns = str(vv[counter][1]) + " less than " + (str(d)) + " less than or equal " + str(vv[counter][2])
            c.append(columns)
            columns = (str(d)) + " greater than " + str(vv[counter][2])

            c.append(columns)
            counter += 1
        # Construct a dataframe with the column names indicate the bins
        df = pd.DataFrame(0, index=np.arange(X_bins_.shape[0]), columns=c)

        return df, c

    def encode_bins_num(self, X_bins_combinded, c):
        X_bins = X_bins_combinded
        ar = np.zeros((len(X_bins), len(c)))

        counter = 0  # goes over patients

        for ind in X_bins:
            counter1 = 0
            for value in X_bins[ind]:
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

        df_num_bins = pd.DataFrame(ar, index=np.arange(len(X_bins)), columns=c)

        return df_num_bins

class Preprocessing_ADNI:
    def __init__(self):
        self.encoder = None
    def encoding_ADNI(self, X, encoder=None):
        # Categorical columns
        X_cate = X[['APOE4', 'DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY']].astype("category").copy()

        encoder = OneHotEncoderMissing(handle_unknown="ignore", sparse=False)
        # X_cate is an array after fit_transform encoding
        X_cate = encoder.fit_transform(X_cate)

        X_ca_encoded = pd.DataFrame(X_cate)

        return X_ca_encoded, encoder

    # @staticmethod
    def preprocessing_ADNI(self, X):
        # Choose which columns should be discretized
        X_num = X[['AGE', 'PTEDUCAT', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'LDELTOTAL', 'MMSE',
                   'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV']]

        # Only not nan values can be discretized
        X_bins_, bin_edges = Binning().form_bins(X_num)

        df, c = Binning().create_quantile_df_num(bin_edges, X_bins_)

        # discretized dataframe
        df_num_bins = Binning().encode_bins_num(X_bins_, c)

        # remove original column names
        X_ = X.drop(['AGE', 'PTEDUCAT', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'LDELTOTAL', 'MMSE',
                     'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal', 'Fusiform', 'ICV'], axis=1).copy()

        df_num_bins.index = X_.index

        X_ca_encoded, encoder = self.encoding_ADNI(X_)

        # Combine continous columns in bins and one-hot encoded categorical columns
        X_discretized = pd.concat([df_num_bins, X_ca_encoded], axis=1)  # X_discretized is a dataframe

        return X_discretized
