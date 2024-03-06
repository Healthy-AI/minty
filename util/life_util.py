import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from util.missingness import Missingness
from sklearn.preprocessing import LabelEncoder

class preprocessing_life:
    '''
        X_df = X [n, d], raw data
        X = X [n, d], preprocessed in bins
        Xm = X with missing values [n, d]
        y = y [n, 1] Outcome variable normalized
        m = missingness mask [n, d]
        '''

    def __init__(self, m=None):
        self.m = m

    '''def missingness(self, X_df):
        return M '''

    def get_data(self):
        data = pd.read_csv(r'../data/datasets/Life-Expectancy_non_discretized.csv')
        life = data[data['Country'].notna()]

        X_df = life.loc[:, life.columns != 'Life_expectancy']

        # Outcome variable
        Y = life['Life_expectancy'].values.reshape(-1, 1)
        std_of_Y = Y.std()
        print('This is the std of Y', std_of_Y)

        #X_df = X_df.drop(columns=['Country'])
        cols = X_df.columns

        #all numeric  features
        num_cols_all = X_df._get_numeric_data().columns

        #special columns
        spec_cols = ['Economy_status_Developed', 'Economy_status_Developing']
        # categorical features
        cat_cols = ['Region', 'Country']
        #numeric features
        num_cols = list(set(num_cols_all) - set(spec_cols))

        # add missingness MCAR
        X_miss, M = Missingness().get_missingness(X_df, p=0.1)

        # add missingness MNAR or MAR
        #X_miss, M = Missingness().add_missingness(X_df, cat_cols, num_cols_all)
        #add customised missingness
        # X_miss, _ = Missingness().get_missingness(X_df, p=0.1)

        return X_miss, Y, spec_cols, num_cols, cat_cols


    def form_bins_num(self, X_miss, num_cols):
        num_X = X_miss[num_cols].copy()
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


    def create_columnnames_num(self, bin_edges, X_miss, num_cols):
        num_X = X_miss[num_cols].copy()
        new_columns_names = []
        origi_column_names = num_X.columns
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
        X_bins_, bin_edges = self.form_bins_num(X_miss, num_cols)
        new_columns_names = self.create_columnnames_num(bin_edges, X_miss, num_cols)
        df_num_bins = self.create_df_num(X_bins_, new_columns_names)

        return df_num_bins

    def get_cat_dummies(self, X_miss, cat_cols):
        X_cat = X_miss[cat_cols].copy()
        cat_dummies = pd.get_dummies(X_cat, drop_first=True, dummy_na=True)

        for c in cat_cols:
            cc = f'{c}_nan'
            if cc in cat_dummies.columns:
                # Replace all 1s with np.nan (or consider leaving as pd.NA if that's acceptable for your use case)
                cat_dummies.loc[cat_dummies[cc] == 1, cat_dummies.columns != cc] = np.nan
                cat_dummies.drop(columns=[cc], inplace=True)

        # Convert the DataFrame to float, then back to int to avoid TypeError when NaNs are present
        cat_dummies = cat_dummies.fillna(-1).astype(int)  # Temporarily replace np.nan with -1
        cat_dummies = cat_dummies.replace(-1, np.nan)  # Replace -1 back with np.nan

        print("These are the categorical binarized columns", cat_dummies, cat_dummies.columns)
        return cat_dummies

    def create_binarized_df(self, X_miss, cat_cols, num_cols, spec_cols):
        # discretize numeric features with missingness
        df_num_bins = self.discretize_num(X_miss, num_cols)
        #Onehotencode categorical features 
        df_cat_dummies = self.get_cat_dummies(X_miss, cat_cols)
        #combine all dataframes to one
        X_m = pd.concat([df_num_bins, X_miss[spec_cols], df_cat_dummies], axis=1)
        
        print('This is X_m binarized',X_m.head(10),  X_m.columns)
        return X_m
