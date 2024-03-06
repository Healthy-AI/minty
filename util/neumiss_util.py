import torch
from torch.utils.data import TensorDataset
import numpy as np

class Preporcess_neumiss:
    #This function give a warning since torch.tensor is not recommended
    def transform_to_tensor(self, X_train, X_test, X_val, y_train, y_test, y_val):
            #Check y_train shape
        y_train = y_train.flatten()
        y_val = y_val.flatten()
        y_test = y_test.flatten()

        y_traintorch = torch.from_numpy(y_train)
        y_valtorch = torch.from_numpy(y_val)
        y_testtorch = torch.from_numpy(y_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_val = X_val.astype(np.float32)
        # Convert to PyTorch TensorDataset
        ds_train = TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_traintorch, dtype=torch.float)
        )

        ds_val = TensorDataset(
            torch.tensor(X_val, dtype=torch.float),
            torch.tensor(y_valtorch, dtype=torch.float)
        )

        ds_test = TensorDataset(
            torch.tensor(X_test, dtype=torch.float),
            torch.tensor(y_testtorch, dtype=torch.float)
        )

        return ds_train, ds_val, ds_test