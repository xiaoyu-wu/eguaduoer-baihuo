import numpy as np
import datetime

def moving_average_pred(train_data, memory_length=100):
    X = train_data.values
    y = X[memory_length:, 0:2]
    y_pred = np.copy(y)
    start_date = X[memory_length, 0]
    stop_date = X[-1, 0]
    for i in range(y_pred.shape[0]):
        memory_start = y_pred[i, 0] - datetime.timedelta(days=memory_length)
        memory_stop = y_pred[i, 0] - datetime.timedelta(days=1)
        a = (X[:, 0] >= memory_start)
        b = (X[:, 0] <= memory_stop)
        c = np.multiply(a, b)
        condlist = [c]
        choicelist = [X[:, 1]]
        select = np.select(condlist, choicelist)
        y_pred[i, 1] = select.sum() / condlist[0].sum()
    return y, y_pred
