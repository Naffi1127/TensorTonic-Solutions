import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    pass
    import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # 1. Get the number of samples (N)
    n_samples = y_true.shape[0]
    
    # 2. Extract the predicted probabilities for the actual true classes
    # We use integer array indexing: y_pred[row_indices, col_indices]
    true_class_probs = y_pred[np.arange(n_samples), y_true]
    
    # 3. Compute the log of those probabilities
    log_probs = np.log(true_class_probs)
    
    # 4. Return the negative average
    return -np.mean(log_probs)
    import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Convert lists to numpy arrays first
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Now you can safely use .shape or len()
    n_samples = y_true.shape[0]
    
    # Advanced indexing to pick the predicted prob for each true class
    true_class_probs = y_pred[np.arange(n_samples), y_true]
    
    # Calculate negative log likelihood and take the mean
    return -np.mean(np.log(true_class_probs))
    