import datetime
import pandas as pd
import numpy as np

def fetch_item_data_by_store(input_dataframe, item_nbr, store_nbr):
    """Fetches single item data from a dask dataframe by store_nbr
    """
    idx = (input_dataframe['store_nbr'] == store_nbr) & (input_dataframe['item_nbr'] == item_nbr)
    selected_portion = input_dataframe.loc[idx, ['date', 'unit_sales', 'onpromotion']]
    selected_portion_comp = selected_portion.compute()
    selected_portion_comp['date'] = pd.to_datetime(selected_portion_comp['date'])
    return selected_portion_comp

def impute_item_data(single_item_data, start_date=datetime.date(2013, 1, 1), end_date=datetime.date(2017, 8, 15)):
    """Imputes the raw data so that dates are continuous
    """
    item_sales_time_series = single_item_data['unit_sales']
    item_sales_time_series.index = single_item_data['date']
    num_days = int((end_date - start_date) / datetime.timedelta(days=1) + 1)
    full_date_list = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    full_time_series = pd.Series([0] * num_days, index=full_date_list)
    for i in full_date_list:
        try:
            full_time_series[i] = item_sales_time_series[i]
        except:
            pass
    return full_time_series


def generate_examples(single_item_full_history, memory_length=100, prediction_scope=5, example_spacing=10):
    """Generates training set with target from imputed full-date time sereies of single item
    """
    start_dates = np.arange(0, len(single_item_full_history) - prediction_scope - memory_length + 1, example_spacing)
    m = len(start_dates)
    n = memory_length
    X = np.zeros([m, n])
    y = np.zeros([m, prediction_scope])
    for i in range(m):
        X[i, :] = single_item_full_history.values[start_dates[i] : start_dates[i] + memory_length]
        y[i, :] = single_item_full_history.values[start_dates[i] + memory_length : start_dates[i] + memory_length + prediction_scope]
    return X, y
