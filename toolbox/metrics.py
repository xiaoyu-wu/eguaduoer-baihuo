import numpy as np

def NWRMSLE(y, y_pred, w):
    """Normalized Weighted Root Mean Squared Logarithmic Error
    """
    return np.sqrt(w.dot((np.log(y_pred + 1) - np.log(y + 1)) ** 2) / w.sum())


if __name__ == "__main__":
    # Test for the NWRMSLE function
    y1 = np.array([0, 0, 0])
    y1_pred = np.array([0, 0, 0])
    w = np.array([1, 1, 2])
    assert NWRMSLE(y1, y1_pred, w) == 0
    y2 = np.array([0, 0, 0])
    y2_pred = np.array([0, 0, (np.e - 1)])
    assert NWRMSLE(y2, y2_pred, w) == np.sqrt(0.5)
    y3 = np.array([0, 0, 0])
    y3_pred = np.array([0, (np.e - 1), (np.e - 1)])
    assert NWRMSLE(y3, y3_pred, w) == np.sqrt(0.75)
