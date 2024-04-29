import numpy as np

def confusion_matrix(y_true, y_pred):
    
    TN = sum((y_true == 0) & (y_pred == 0))
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
   
    return np.matrix([[TN, FP], [FN, TP]])