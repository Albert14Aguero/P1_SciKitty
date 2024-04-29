def f1_score(y_test, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_test, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_test, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_test, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1