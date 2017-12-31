# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import time
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Any results you write to the current directory are saved as output.
start_time = time.time()

X_train = pd.read_csv('../input/X_train.csv')
y_train = np.genfromtxt('../input/y_train.csv', delimiter=",")
X_val = pd.read_csv('../input/X_val.csv')
y_val = np.genfromtxt('../input/y_val.csv', delimiter=",")
X_test = pd.read_csv('../input/X_test.csv')

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 101688780)  # 2017-01-01
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

df_2017 = df_train[df_train.date.isin(
    pd.date_range("2017-01-04", periods=7 * 32))].copy()
del df_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

# Use small portion of data for fast development
FAST_DEV = False
if FAST_DEV:
    DEV_PERCENT = 0.30
    print("Train with {}% data.".format(DEV_PERCENT*100))
    m_train = int(len(X_train) * DEV_PERCENT)
    X_train = X_train.iloc[:m_train, :]
    y_train = y_train[:m_train, :]
    # m_val = int(len(X_val) * DEV_PERCENT)
    # X_val = X_val.iloc[:m_val, :]
    # y_val = y_val[:m_val, :]
else:
    DEV_PERCENT = 1
    print("Train with {}% data.".format(DEV_PERCENT*100))


print("Training and predicting models...")
params = {
    'num_leaves': 2**8 - 1,
    'objective': 'regression_l2',
    # 'max_depth': 8,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.50,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 8
}

cate_names = ['store_type', 'store_cluster', 'item_perishable', 'item_class']
cate_vars = [list(X_train.columns).index(i) for i in cate_names]
print("{} categorical features found in the training set. column #: {}".format(len(cate_vars), cate_vars))

MAX_ROUNDS = 3000
val_pred = []
test_pred = []

for i in range(16):
    print("=" * 50)
    print("Prediction for Day %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=X_train["item_perishable"] * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=X_val["item_perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('../output/lgb.csv', float_format='%.4f', index=None)

print("------ Time used: {} s ------".format(time.time()-start_time))
