import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    pass
    import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Returns (param_new, m_new, v_new).
    """
    # Convert to numpy arrays to ensure vectorized operations
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # 1. Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad
    
    # 2. Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    
    # 3. Compute bias-corrected first moment estimate
    m_hat = m_new / (1 - beta1**t)
    
    # 4. Compute bias-corrected second raw moment estimate
    v_hat = v_new / (1 - beta2**t)
    
    # 5. Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new
    