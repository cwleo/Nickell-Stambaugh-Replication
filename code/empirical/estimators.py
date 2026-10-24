import numpy as np


def within_transform(matrix):
    return matrix - np.mean(matrix, axis=1, keepdims=True)

def z_estimator(rho_z, x):
    N, T = x.shape
    z = np.zeros((N, T))
    delta_x = np.zeros((N, T))
    delta_x[:, 0] = x[:, 0]
    delta_x[:, 1:]=x[:, 1:] - x[:, :-1]
    for t in range(T):
        weights = rho_z ** np.arange(t,-1,-1)
        z[:, t] = np.sum(delta_x[:, :t+1]*weights, axis=1)
    return z

def ivx_estimator(rho_z, x, y):    
    z = z_estimator(rho_z, x)   
    y_lead = np.roll(y, -1, axis=1)[:, :-1]  # t=2,...,T
    z_trim = z[:, :-1]
    x_trim = x[:, :-1]
    z_tilde = within_transform(z_trim)
    
    beta_ivx = np.sum(z_tilde * y_lead) / np.sum(z_tilde * x_trim)
    return beta_ivx

def beta_WG_estimator(x,y):
    y_lead = np.roll(y, -1, axis=1)[:, :-1]  # t=2,...,T
    x_trim = x[:, :-1]
    x_tilde = within_transform(x_trim)
    
    beta_WG = np.sum(x_tilde * y_lead) / np.sum(x_tilde ** 2)
    return beta_WG

def rho_WG_estimator(x):
    x_lead = np.roll(x, -1, axis=1)[:, :-1]
    x_trim = x[:, :-1]
    x_tilde = within_transform(x_trim)
    
    rho_WG = np.sum(x_tilde * x_lead) / np.sum(x_tilde ** 2)
    return rho_WG

def rho_ivx_estimator(rho_z2, T, x):
    N, T = x.shape
    G = int(np.floor(T**0.25))
    z = np.zeros((N, T))
    delta_x = np.zeros((N, T))
    delta_x[:, 0] = x[:, 0]
    delta_x[:, 1:]=x[:, 1:] - x[:, :-1]
    z[:, 0] = delta_x[:, 0]
    for t in range(1, T):
        z[:, t] = rho_z2 * z[:, t-1] + delta_x[:, t]

    rho_WG = rho_WG_estimator(x)
    v = np.zeros((N, T))
    v[:, 1:] = within_transform(x[:, 1:]) - rho_WG * within_transform(x[:, :-1])
    kappa_sum = 0.0
    for h in range(1, G+1):
        kappa_h = np.sum(v[:, :T-h] * v[:, h:]) / (N * (T - h))
        kappa_sum += kappa_h
    rho_ivx = (np.sum(z[:, :-1] * x[:, 1:])-kappa_sum*N*(T-1))/np.sum(z[:, :-1] * x[:, :-1])
    if rho_ivx < -0.99:
        rho_ivx = -0.99
    return rho_ivx

def covariance_omegah(x, y, rho_estimator, beta_estimator, h):
    N, T = x.shape
    x_tilde1 = within_transform(x[:, 1+h:T])
    x_tilde2 = within_transform(x[:, h:T-1])
    v_hat = x_tilde1 - rho_estimator * x_tilde2
    y_tilde = within_transform(y[:, 1:T-h])
    x_tilde3 = within_transform(x[:, :T-h-1])
    e_hat = y_tilde - beta_estimator * x_tilde3
    omega_h = np.sum(v_hat * e_hat)/(N * (T-h))
    return omega_h

def Psi_h(rho_star, rho_z, T, h):
    # case1: ρ* = 1 但 ρ_z ≠ 1
    if np.isclose(rho_star, 1) and not np.isclose(rho_z, 1):
        numerator = rho_z - rho_z**(T - h)
        denominator = 1 - rho_z
        return (1 / (rho_z - 1)) * (numerator / denominator - (T - h - 1))
    
    # case2: ρ* ≠ 1 但 ρ_z = 1
    elif not np.isclose(rho_star, 1) and np.isclose(rho_z, 1):
        numerator = rho_star - rho_star**(T - h)
        denominator = 1 - rho_star
        return (1 / denominator) * ((T - h - 1) - numerator / denominator)

    #case3: ρ* ≠ 1 且 ρ_z ≠ 1
    elif not np.isclose(rho_star, 1) and not np.isclose(rho_z, 1):
        term1 = (rho_z - rho_z ** (T - h)) / (1 - rho_z)
        term2 = (rho_star - rho_star ** (T -h)) / (1 - rho_star)
        return (term1 - term2) / (rho_z - rho_star) 
    
    # case4: ρ* = 1 且 ρ_z = 1
    elif np.isclose(rho_star, 1) and np.isclose(rho_z, 1):
        return (T - h - 1)*(T-h)/2

def IVX_bias(x, rho_z, rho_star, y):
    N, T = x.shape
    G = int(np.floor(T**0.25))
    Psi_values = np.array([Psi_h(rho_star, rho_z, T, h) for h in range(G+1)])
    beta_ivx = ivx_estimator(rho_z, x, y)
    omega_h = np.array([covariance_omegah(x, y, rho_star, beta_ivx, h) for h in range(G+1)])
    numerator = N * np.sum(Psi_values * omega_h)
    z_tilde = within_transform(z_estimator(rho_z, x))
    denominator = T * np.sum(z_tilde * x)
    return numerator / denominator

def ivx_debiased(rho_z, x, y, rho_star):
    return ivx_estimator(rho_z, x, y)+IVX_bias(x, rho_z, rho_star, y)

def WG_bias(x, rho_z, rho_star, y):
    N, T = x.shape
    G = int(np.floor(T**0.25))
    Psi_values = np.array([Psi_h(rho_star, rho_z, T, h) for h in range(G+1)])
    beta_WG = beta_WG_estimator(x,y)
    omega_h = np.array([covariance_omegah(x, y, rho_star,beta_WG, h) for h in range(G+1)])
    numerator = N * np.sum(Psi_values * omega_h)
    x_tilde = within_transform(x)
    denominator = (T-1) * np.sum(x_tilde ** 2)
    return numerator / denominator

def WG_debiased_estimator(rho_z, x, y, rho_star):
    return beta_WG_estimator(x,y)+ WG_bias(x, rho_z, rho_star, y)

def IVX_se_pvalue(x, y, z, rho, beta, z_tilde, beta_true_value):
    N, T = x.shape
    denominator = np.sum(z_tilde * x)
    G = int(np.floor(T**0.25))
    omega_ev = np.zeros(G+1)
    omega_v = np.zeros(G+1)
    e_hat = np.zeros((N, T))
    e_hat[:, 1:] = within_transform(y[:, 1:]) - beta * within_transform(x[:, :-1])
    v_hat = np.zeros((N, T))
    v_hat[:, 1:] = within_transform(x[:, 1:]) - rho * within_transform(x[:, :-1])
    for h in range(G+1):
        omega_ev[h] = np.sum(e_hat[:,:T-h] * v_hat[:,h:])/(N*(T-h))
        
    a = np.sum(omega_ev)**2/(np.sum(e_hat**2)/(N*T))
    term1 = np.sum((z[:, :-1])**2 *(e_hat[:,1:]**2))
    term2 = np.maximum(1-a* (N/(T**1.5)), 0)*T*np.sum(np.mean(z, axis=1)**2)
    term3 = np.sum((z_tilde[:, :-1])**2 *(e_hat[:,1:]**2))
    w_11 = np.sqrt(np.sum(e_hat**2)/(N*(T-2)))
    if term1 - term2* (w_11**2) >=0:
        se = np.sqrt(term1 -(w_11**2) * term2)/denominator    
    else:
        se = np.sqrt(term3  )/denominator

    from scipy.stats import norm
    t_stat = (beta - beta_true_value) / se
    p_value = 2 * norm.sf(np.abs(t_stat))
    
    return t_stat, se, p_value

def WG_se_pvalue(x, y, rho, beta, x_tilde, beta_true_value):
#     N, T = x.shape
    e_hat = within_transform(y[:, 1:]) - beta * within_transform(x[:, :-1])
    denominator = np.sum(x_tilde ** 2)
    term1 = np.sum(x_tilde[:, :-1]**2 * (e_hat**2))
    term2 = 0
    if term1 - term2 >=0:
        se = np.sqrt(term1-term2)/denominator 
    else:
        se = np.sqrt(term1)/denominator

    from scipy.stats import norm
    t_stat = (beta - beta_true_value) / se
    p_value = 2 * norm.sf(np.abs(t_stat))
    
    return t_stat, se, p_value
