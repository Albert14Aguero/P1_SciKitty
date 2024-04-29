def precision(y_true, y_pred):

    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
   
    return TP / (TP + FP) if (TP + FP) != 0 else 0