import numpy as np
import EntropyHub as EH

def coefficient_of_variation(x):
    # Coefficient of variation (CV) = standard deviation / mean
    return x.std() / x.mean()


def average_real_variability(data):
    """
    Compute the Average Real Variability (ARV) of a given dataset - sum of absolute differences between consecutive values

    Parameters:
    data (list or array-like): A list or array of numerical observations.

    Returns:
    float: The computed ARV.
    """
    # if pandas series, convert to numpy array
    if 'pandas' in str(type(data)):
        data = data.values

    if len(data) < 2:
        return np.nan

    n = len(data)
    differences = [abs(data[i] - data[i - 1]) for i in range(1, n)]
    arv = sum(differences) / (n - 1)
    return arv

def complexity_index(data, embedding_dimension=2, scales=2, F_Order=3, F_Num=0.5, RadNew=4, Plotx=False):
    # if pandas series, convert to numpy array
    if 'pandas' in str(type(data)):
        data = data.values

    # embedding dimension m and tolerance r are in most cases assumed to be 2 and 0.2*signal SD (standard deviation)
    r = 0.2 * np.std(data)
    Mobj = EH.MSobject('SampEn', m=embedding_dimension, r=r)

    try:
        MSx, Ci = EH.rMSEn(data, Mobj, Scales=scales, F_Order=F_Order, F_Num=F_Num, RadNew=RadNew, Plotx=Plotx)
        return Ci
    except:
        return np.nan
