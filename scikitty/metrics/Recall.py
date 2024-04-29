def recall(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    
    # Calcular el recall
    return TP / (TP + FN) if (TP + FN) != 0 else 0