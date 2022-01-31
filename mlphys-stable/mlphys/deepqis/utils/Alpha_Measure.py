def alpha(D, K, mean):
    alpha_x = D * (mean - 1) / (D + K - 1 - mean * K * D)
    return alpha_x
