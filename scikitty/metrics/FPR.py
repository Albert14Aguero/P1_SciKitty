def fpr(y_true, y_pred):
    
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
  
    return FP / (FP + TN) if (FP + TN) != 0 else 0