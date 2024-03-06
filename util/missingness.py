import pandas as pd
import numpy as np
import torch
from util.utils import *
from sklearn.preprocessing import LabelEncoder

class Missingness:
    """
    Generate missingness mask for data"

    todo: provide code for different missingness mechanisms: MAR, MNAR
    customized missingness for different datasets
    or at least missingness for correlated features depending on each data set
    """

    #MCAR
    def get_missingness(self, data, p):
        m = data.shape[0]  # samples
        d = data.shape[1]  # no. of covariates

        M = np.random.rand(m, d) < p
        X_miss = data.copy()
        X_miss[M == 1] = np.nan

        return X_miss, M


    def add_missingness(self, X_df, cat_cols, num_cols_all):
        num_X_float = X_df[num_cols_all].apply(pd.to_numeric)

        #label encode categorical features
        X_df_cat_encoded = X_df[cat_cols].copy()
        label_encoders = {}

        # Loop through categorical columns and create LabelEncoder for each
        for col in cat_cols:
            le = LabelEncoder()
            X_df_cat_encoded[col] = le.fit_transform(X_df_cat_encoded[col])
            label_encoders[col] = le

        #convert all columns categorcial label encoded columns to float
        X_df_cat_encoded = X_df_cat_encoded.apply(pd.to_numeric)
        #merge numerical and categorical features in float to numpy array
        X_labelencoded_num = pd.concat([num_X_float, X_df_cat_encoded], axis=1).to_numpy()

        #choose mechansim to be 'MAR' or 'MNAR', input musst be a numpy array
        #X_nas, M = Missingness().produce_NA(X_labelencoded_num, 0.1, "MAR", 'logistic',0.1, 0.01)
        X_nas, M = Missingness().find_correlated_pairs(X_labelencoded_num, 0.7)

        # Convert back to pandas DataFrame
        X_miss = pd.DataFrame(X_nas, index=X_df.index, columns=X_df.columns)

        # Reconstruct original values for label-encoded columns
        X_miss_cat = X_miss[cat_cols].copy()
        X_miss_cat = X_miss_cat.apply(pd.to_numeric)
        for col in cat_cols:
            X_miss_cat[col] = X_miss_cat[col].apply(lambda x: self.reconstruct_categorical_values(x, label_encoders[col]))

        #Replace all label encoded columns with categorical columns
        X_miss[X_miss_cat.columns] = X_miss_cat

        M = pd.DataFrame(M, index=X_df.index, columns=X_df.columns)
        X_miss[M == 1] = np.nan

        return X_miss, M



    def encode_categorical_columns(self, df, categorical_columns):
        """
        Encode categorical columns in a DataFrame to numerical values using LabelEncoder.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_columns (list): List of column names with categorical values to be encoded.

        Returns:
        pd.DataFrame: A new DataFrame with categorical columns transformed to numerical values.
        """

        df_encoded = df.copy()  # Create a copy of the input DataFrame to avoid modifying the original.

        label_encoder = LabelEncoder()

        for column in categorical_columns:
            if column in df_encoded.columns:
                df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

        return df_encoded

    def process_array(self, X):
        # Check if the input is already a NumPy array
        if not isinstance(X, np.ndarray):
            try:
                # Try to convert the input to a NumPy array
                X = np.asarray(X)
            except Exception as e:
                # Handle any exceptions that may occur during conversion
                print(f"Failed to convert input to NumPy array: {e}")

        # Now 'X' is guaranteed to be a NumPy array (either originally or after conversion)
        # You can proceed with processing 'input_array' here

        return X

    def produce_NA(self, X, p_miss, mecha="MAR", opt=None, p_obs=None, q=None):
        """
        Generate missing values for specifics missing-data mechanism and proportion of missing values.

        Parameters
        ----------
        X : torch.DoubleTensor or np.ndarray, shape (n, d)
            Data for which missing values can be simulated or used from data set
            If a numpy array is provided, it will be converted to a pytorch tensor.
        p_miss : float
            Proportion of missing values to generate for variables which will have missing values.
        mecha : str,
                Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
        opt: str,
             For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
        p_obs : float
                If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
        q : float
            If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

        Returns
        ----------
        A dictionnary containing:
        'X_init': the initial data matrix.
        'X_incomp': the data with the generated missing values.
        'mask': a matrix indexing the generated missing values.
        """
        X = self.process_array(X)

        to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = X.astype(np.float32)
            X = torch.from_numpy(X)

        # Decide on the missingness mechanism
        if mecha == "MAR":
            mask = MAR_mask(X, p_miss, p_obs).double()
        elif mecha == "MNAR" and opt == "logistic":
            mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
        elif mecha == "MNAR" and opt == "quantile":
            mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
        elif mecha == "MNAR" and opt == "selfmasked":
            mask = MNAR_self_mask_logistic(X, p_miss).double()
        else:
            mask = (torch.rand(X.shape) < p_miss).double()

        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        X_nas = X_nas.numpy()

        #return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}
        return X_nas, mask


