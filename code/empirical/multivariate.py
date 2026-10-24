import numpy as np


def z_estimator_multiple(rho_z, x):
    K, N, T = x.shape
    z = np.zeros((K, N, T))
    delta_x = np.zeros((K, N, T))
    
    delta_x[:, :, 0] = x[:, :, 0]  
    delta_x[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    
    for j in range(K):
        for t in range(T):
            weights = rho_z ** np.arange(t, -1, -1)  
            z[j, :, t] = np.sum(delta_x[j, :, :t+1] * weights, axis=1)
    
    return z

def within_transform(matrix):
    return matrix - np.mean(matrix, axis=1, keepdims=True)

def within_transform_multiple(matrix):
    return matrix - np.mean(matrix, axis=2, keepdims=True)

def ivx_estimator_multiple(rho_z, x, y):
    K, N, T = x.shape
    z = z_estimator_multiple(rho_z, x)
    z_tilde = within_transform_multiple(z[:,:,:-1])  # (K, N, T-1)
    y_lead = np.roll(y, -1, axis=1)[:, :-1]  # (N, T-1)
    
    num = np.sum(z_tilde * y_lead, axis=(1, 2))
    denominator = np.einsum('jnt,knt->jk', z_tilde, x[:, :, :-1])
    try:
        beta_ivx = np.linalg.solve(denominator, num)#beta_ivx = denominator_inv @ num 
    except np.linalg.LinAlgError:
        print("Warning: Denominator matrix is singular, returning zero vector.")
        beta_ivx = np.zeros(K)
    
    return beta_ivx

def rho_WG_estimator_multiple(x):
    K, N, T = x.shape
    rho_WG = np.zeros(K)
    x1 = within_transform_multiple(x[:, :, :-1])
    
    for j in range(K):
        x_j = x[j, :, :] 
        x_tilde = x1[j, :, :] 
        x_lead = np.roll(x_j, -1, axis=1)[:, :-1]  
        num = np.sum(x_tilde * x_lead)  
        denom = np.sum(x_tilde ** 2)  
        rho_WG[j] = num / denom 
    
    return rho_WG

def rho_ivx_estimator_multiple(rho_z2,x):
    K, N, T = x.shape
    G = int(np.floor(T**0.25))
    rho_ivx = np.zeros(K)
    rho_WG = rho_WG_estimator_multiple(x)
    
    for j in range(K):
        x_j = x[j, :, :]
        z = np.zeros((N, T))
        delta_x = np.zeros((N, T))
        delta_x[:, 0] = x_j[:, 0]
        delta_x[:, 1:] = x_j[:, 1:] - x_j[:, :-1]
        z[:, 0] = delta_x[:, 0]
        for t in range(1, T):
            z[:, t] = rho_z2 * z[:, t-1] + delta_x[:, t]
        
        v = np.zeros((N, T))
        v[:, 1:] = within_transform(x_j[:, 1:]) - rho_WG[j] * within_transform(x_j[:, :-1])
        
        kappa_sum = 0.0
        for h in range(1, G+1):
            kappa_h = np.sum(v[:, :T-h] * v[:, h:]) / (N * (T - h))
            kappa_sum += kappa_h
        
        numerator = np.sum(z[:, :-1] * x_j[:, 1:]) - kappa_sum * N * (T - 1)
        denominator = np.sum(z[:, :-1] * x_j[:, :-1])
        rho_ivx[j] = numerator / denominator
        
        if rho_ivx[j] < -0.99:
            rho_ivx[j] = -0.99
    
    return rho_ivx

def Psi_h_multiple(rho_star, rho_z, T, h):
    K = rho_star.shape[0]
    phi_h = np.zeros(K)
    
    for k in range(K):
        # case1: ρ* = 1 , ρ_z ≠ 1
        if np.isclose(rho_star[k], 1) and not np.isclose(rho_z, 1):
            numerator = rho_z - rho_z**(T - h)
            denominator = 1 - rho_z
            phi_h[k] = (1 / (rho_z - 1)) * (numerator / denominator - (T - h - 1))
        
        # case2: ρ* ≠ 1 , ρ_z = 1
        elif not np.isclose(rho_star[k], 1) and np.isclose(rho_z, 1):
            numerator = rho_star[k] - rho_star[k]**(T - h)
            denominator = 1 - rho_star[k]
            phi_h[k] = (1 / denominator) * ((T - h - 1) - numerator / denominator)

        # case3: ρ* ≠ 1 , ρ_z ≠ 1
        elif not np.isclose(rho_star[k], 1) and not np.isclose(rho_z, 1):
            term1 = (rho_z - rho_z ** (T - h)) / (1 - rho_z)
            term2 = (rho_star[k] - rho_star[k] ** (T - h)) / (1 - rho_star[k])
            phi_h[k] = (term1 - term2) / (rho_z - rho_star[k])
        
        # case4: ρ* = 1 , ρ_z = 1
        elif np.isclose(rho_star[k], 1) and np.isclose(rho_z, 1):
            phi_h[k] = (T - h - 1) * (T - h) / 2
    
    return phi_h

def covariance_omegah_multiple(x, y, rho_estimator, beta_estimator, h):
    K, N, T = x.shape
    omega_h = np.zeros(K)
    
    y_tilde = within_transform(y[:, 1:T-h]) #1:T-h ;1+h:T
    x_tilde3 = within_transform_multiple(x[:, :, :T-h-1])#:T-h-1 ;h:T-1 
    e_hat = y_tilde - np.sum(beta_estimator[:, np.newaxis, np.newaxis] * x_tilde3, axis=0)

    for j in range(K):
        x_tilde1 = within_transform(x[j, :, 1+h:T])#1+h:T ;1:T-h
        x_tilde2 = within_transform(x[j, :, h:T-1]) #h:T-1 ;:T-h-1
        v_hat = x_tilde1 - rho_estimator[j] * x_tilde2 
        omega_h[j] = np.sum(v_hat * e_hat) / (N * (T-h-1))  
    
    return omega_h

def IVX_bias_multiple(x, rho_z, rho_star, y):
    K, N, T = x.shape
    G = int(np.floor(T**0.25))
    
    Psi_values = np.array([Psi_h_multiple(rho_star, rho_z, T, h) for h in range(G+1)]).T  # (K, G+1)
    beta_ivx = ivx_estimator_multiple(rho_z, x, y)  # (K,)
    omega_h = np.array([covariance_omegah_multiple(x, y, rho_star, beta_ivx, h) for h in range(G+1)]).T  # (K, G+1)
    numerator = N * np.sum(Psi_values * omega_h, axis=1)/T  # (K,)
    z_tilde = within_transform_multiple(z_estimator_multiple(rho_z, x))  # (K, N, T)
    denominator = np.einsum('jnt,knt->jk', z_tilde, x)
    
    try:
        bias = np.linalg.solve(denominator, numerator)  
    except np.linalg.LinAlgError:
        print("Warning: Denominator matrix is singular, returning zero vector.")
        bias = np.zeros(K)
    
    return bias

def ivx_debiased_multiple(rho_z, x, y, rho_star):
    return ivx_estimator_multiple(rho_z, x, y)+IVX_bias_multiple(x, rho_z, rho_star, y)

def wald_stat(x, y, z, beta, rho, beta_true_value):
    from scipy.stats import chi2
    K, N, T = x.shape
    theta = 0.95
    T_eff = T - 1
    e_hat = np.zeros((N, T))
    e_hat[:, 1:] = within_transform(y[:, 1:]) - np.sum(beta[:, np.newaxis, np.newaxis] * within_transform_multiple(x[:, :, :T_eff]), axis=0)
    z_tilde = within_transform_multiple(z)  # (K, N, T-1)
    denominator = np.einsum('jnt,knt->jk', z_tilde, x)  # (K, K)
    v_hat = np.zeros((K, N, T))
    for j in range(K):
        v_hat[j, :, 1:] = within_transform(x[j, :, 1:]) - rho[j] * within_transform(x[j, :, :-1])
    
    G = int(np.floor(T**0.25))
    omega_ev = np.zeros((K, G+1))
    omega_v = np.zeros((K, G+1))
    for j in range(K):
        for h in range(G+1):
            omega_ev[j, h] = np.sum(e_hat[:, :T-h] * v_hat[j, :, h:]) / (N * (T-h))
            
    numerator_a = np.sum(omega_ev, axis=1) ** 2 
    denominator_a = np.sum(e_hat ** 2) / (N * T)
    a = numerator_a / denominator_a
    diag_values = np.sqrt(np.maximum(1 - a * (N / T**1.5), 0))
    D = np.diag(diag_values) 

    w_11 = np.sum(e_hat** 2) / (N * T)
    term1 = np.einsum('jnt,nt,knt->jk', z[:,:,:-1], e_hat[:,1:]**2, z[:,:,:-1])#w_11 * np.einsum('jnt,knt->jk', z[:,:,:-1], z[:,:,:-1])
    z_matrix = np.einsum('jn,kn->jk', np.mean(z, axis = 2), np.mean(z, axis = 2))
    term2 = w_11 * D @ (T * z_matrix) @ D #w_11 * T * np.einsum('ij,jk,kl->il', D, z_matrix, D)

    #test
    if np.min(np.linalg.eigvals(term1 - w_11 * term2)) > 0:
        numerator = term1-w_11 * term2
    else:
        numerator = np.einsum('jnt,nt,knt->jk', z_tilde[:,:,:-1], e_hat[:,1:]**2, z_tilde[:,:,:-1])
    #numerator = term1
    inv_denom = np.linalg.inv(denominator)
    avar = inv_denom @ numerator @ inv_denom.T
    diag = np.diag(avar)
    se = np.zeros_like(diag)
    
    for i in range(len(diag)):
        if diag[i] >= 0:
            se[i] = np.sqrt(diag[i])
        else:
            term_adj = term1.copy()
            avar_i = inv_denom @ term_adj @ inv_denom.T
            diag_i = np.diag(avar_i)[i]
            se[i] = np.sqrt(diag_i)
    
    t_stats = (beta - beta_true_value) / se
    W_stat = beta.T @ np.linalg.inv(avar) @ beta 
    p_value = 1 - chi2.cdf(W_stat, df=K)
    
    return W_stat, se, t_stats, p_value, avar
