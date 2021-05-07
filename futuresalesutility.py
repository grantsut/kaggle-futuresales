import numpy as np
import pandas as pd

def reduce_mem_usage(df, silent=True, allow_categorical=True, float_dtype="float32"):
    """ 
    Iterates through all the columns of a dataframe and modify the data type
     to reduce memory usage. Can also factorize categorical columns to integer.
    """
    def _downcast_numeric(series, allow_categorical=allow_categorical):
        """
        Downcast a numeric series into either the smallest possible int dtype or specified float.
        """
        if pd.api.types.is_sparse(series.dtype) is True:
            return series
        elif pd.api.types.is_numeric_dtype(series.dtype) is False:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return series
            else:
                if allow_categorical:
                    return series
                else:
                    codes, uniques = series.factorize()
                    series = pd.Series(data=codes, index=series.index)
                    series = _downcast_numeric(series)
                    return series
        else:
            series = pd.to_numeric(series, downcast="integer")
        if pd.api.types.is_float_dtype(series.dtype):
            series = series.astype(float_dtype)
        return series

    if silent is False:
        start_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    if df.ndim == 1:
        df = _downcast_numeric(df)
    else:
        for col in df.columns:
            df.loc[:, col] = _downcast_numeric(df.loc[:,col])
    if silent is False:
        end_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def shrink_mem_new_cols(matrix, oldcols=None, allow_categorical=False):
    # Calls reduce_mem_usage with a list of columns to leave unprocessed
    if oldcols is not None:
        newcols = matrix.columns.difference(oldcols)
    else:
        newcols = matrix.columns
    matrix.loc[:,newcols] = reduce_mem_usage(matrix.loc[:,newcols], allow_categorical=allow_categorical)
    oldcols = matrix.columns
    return matrix, oldcols


def list_if_not(s, dtype=str):
    # Puts a variable in a list if it is not already a list
    if type(s) not in (dtype, list):
        raise TypeError
    if (s != "") & (type(s) is not list):
        s = [s]
    return s


def add_lag_feature(
    dforig, feature_series, grouping_fields, lag, fillna=0, optimize_mem=False, add_suffix=True
):
    """
    Takes the series feature_series, increments date_block_num by lag and joins it to the table
    dforig by merging on date_block_num and grouping_fields. NaNs are optionally filled by the fillna.

    """
    grouping_fields = list_if_not(grouping_fields)
    feature_series = feature_series.copy()  # This is necessary
    if add_suffix:
        feature_name = feature_series.name + "_lag_" + str(lag)
        feature_series.name = feature_name
    else:
        feature_name = feature_series.name
    feature_series = feature_series.reset_index()
    feature_series["date_block_num"] += lag
    # Separate condition for when features are aggregated at the date block level
    # A minimal copy of the dataframe is made to speed up merges, and
    # an index column is inserted to preserve order between the original and minimal dataframe
    if len(grouping_fields) == 0:
        dforig["tempidx"] = list(range(len(dforig)))
        df = dforig.loc[:,["tempidx", "date_block_num"]]
        df = df.merge(feature_series, on=["date_block_num"], how="left")
    else:
        dforig["tempidx"] = list(range(len(dforig)))
        df = dforig.loc[:,["tempidx", "date_block_num"] + grouping_fields]

        df = df.merge(feature_series, on=["date_block_num"] + grouping_fields, how="left")

    if type(fillna) is int or type(fillna) is float:
        df[feature_name] = df[feature_name].fillna(fillna)
    elif fillna is None:
        pass
    else:
        raise ValueError
    # Sort dataframes on index field to ensure items match
    df = df.sort_values(by="tempidx")
    dforig = dforig.sort_values(by="tempidx")
    dforig = dforig.drop(columns=["tempidx"])
    # Add lagged feature back to dataframe
    if optimize_mem is True:
        dforig.loc[:,feature_name] = reduce_mem_usage(df[feature_name].to_numpy())
    else:
        dforig.loc[:,feature_name] = df.loc[:,feature_name].to_numpy()

    return dforig


def apply_lags(df, feature_series, grouping_fields, lags, fillna=0):
    # Joins a series to df by multiple lag values
    feat_name = feature_series.name
    print(f"Applying lags for {feat_name}")
    for lag in lags:
        df = add_lag_feature(df, feature_series, grouping_fields, lag, fillna=fillna)
    return df