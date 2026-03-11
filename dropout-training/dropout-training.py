import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    pass
    import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Returns (output, dropout_pattern).
    """
    x = np.array(x)
    
    # 1. Generate random values using the provided rng or np.random
    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)
        
    # 2. Determine the mask (dropout_pattern)
    # To drop with probability 'p', we keep with probability '1 - p'
    # Use strict inequality (<) as most platforms expect this for specific seeds
    keep_prob = 1 - p
    dropout_pattern = (random_values < keep_prob).astype(int)
    
    # 3. Apply the mask and scale (Inverted Dropout)
    # The scale factor 1/(1-p) ensures the expected value remains constant
    scale = 1 / (1 - p)
    output = x * dropout_pattern * scale
    
    return output, dropout_pattern
    import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x)
    
    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)
        
    keep_prob = 1 - p
    # Create the 0/1 mask
    mask = (random_values < keep_prob).astype(float)
    
    # Inverted scaling factor
    scale = 1 / (1 - p)
    
    # The actual output
    output = x * mask * scale
    
    # Based on your "Expected" screenshot, they might want the 
    # dropout_pattern to be the scaled mask (e.g., 0.0 and 2.0)
    dropout_pattern = mask * scale
    
    return output, dropout_pattern
    
    