import pandas as pd
import numpy as np

def normalize_data(data, label_col = None, scale = 'normalize'):
    """This function prepares the data for further processing, splitting the labels 
       (if provided) from the data and normalizing or standardizing the data

    Parameters
    ----------
    data : a pandas dataframe
        The shape is (n_samples, n_features + 1) if labels
        are provided in the dataframe or (n_samples, n_features) if labels are not
        provided

    label_col : int
        Which column index contains the data labels. Select None if no labels are 
        provided in the dataframe (default is None)
    """

    data = data.copy()
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64':
            categorical_cols.append(col)
            data[col] = pd.Categorical(data[col]).codes


    if label_col is not None:
        label = data.columns[label_col]
        y     = data.pop(label)
        x     = data

    else:
        x = data


    if scale == 'standardize':
        for col in x.columns:
            if data[col].std() !=0:
                data[col] = (data[col] - data[col].mean()) / data[col].std()

    elif scale == 'normalize':
        for col in x.columns:
            if data[col].max() != data[col].min():
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    if label_col is None:
        return np.array(x)
    else:
        return np.array(x), y
