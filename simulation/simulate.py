import numpy as np



def sample_nb_r_theta(r, theta, size, rng=None):
    """
    Sample NB with pmf: C(r+y-1, y) * theta^y * (1-theta)^r
    where r can be non-integer.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not (0 < theta < 1):
        raise ValueError("theta must be in (0,1)")
    if r <= 0:
        raise ValueError("r must be > 0")
    scale = theta / (1.0 - theta)          # Gamma scale
    lam = rng.gamma(shape=r, scale=scale, size=size)  # Poisson rate
    return rng.poisson(lam)


# --------------------------
# 1) calculate c_t and A_t
#   c_t = E(exp(-Y_t)) = ((1-theta)/(1-theta*e^-1))^r
#   A_t = r*theta*e^-1/(1-theta*e^-1) - r*theta/(1-theta)
# --------------------------
def compute_c_A(r, theta):
    e_m1 = np.exp(-1.0)
    c = ((1.0 - theta) / (1.0 - theta * e_m1)) ** r
    A = r * theta * e_m1 / (1.0 - theta * e_m1) - r * theta / (1.0 - theta)
    return c, A

# --------------------------
# 3) bivariate NB (Sarmanov) sampling：rejection sampling
#   joint = f1*f2*(1 + lambda*g1*g2),  g(y)=exp(-y)-c
#   joint >=0，lambda must be bounded：
#   |lambda| <= 1/(a1*a2)
#   g(y) in [-c, 1-c] => max|g| = max(c, 1-c) = a
# --------------------------
def bivariate_nb_sarmanov_r_theta(
    n,
    r1, theta1,
    r2, theta2,
    lam,
    rng=None,
    max_iter=10_000_000
):
    rng = np.random.default_rng() if rng is None else rng

    c1, _ = compute_c_A(r1, theta1)
    c2, _ = compute_c_A(r2, theta2)

    a1 = max(c1, 1.0 - c1)
    a2 = max(c2, 1.0 - c2)

    lam_max = 1.0 / (a1 * a2)  
    if abs(lam) > lam_max + 1e-12:
        raise ValueError(f"|lambda| too large. Need |lambda| <= {lam_max:.6g}, got {lam}")

    # Rejection upper bound:
    # w(y1,y2)=1+lam*g1*g2 ∈ [1-|lam|a1a2, 1+|lam|a1a2]
    # so M = 1+|lam|a1a2
    M = 1.0 + abs(lam) * a1 * a2

    y1 = np.empty(n, dtype=int)
    y2 = np.empty(n, dtype=int)

    filled = 0
    trials = 0
    batch = max(1024, n)

    while filled < n:
        if trials > max_iter:
            raise RuntimeError("Rejection sampling hit max_iter; try smaller |lambda| or adjust params.")

        # propose independent NB draws
        prop1 = sample_nb_r_theta(r1, theta1, size=batch, rng=rng)
        prop2 = sample_nb_r_theta(r2, theta2, size=batch, rng=rng)

        g1 = np.exp(-prop1) - c1
        g2 = np.exp(-prop2) - c2
        w = 1.0 + lam * g1 * g2

        # accept with prob w/M
        u = rng.random(batch)
        acc = u < (w / M)

        k = min(np.sum(acc), n - filled)
        if k > 0:
            idx = np.flatnonzero(acc)[:k]
            y1[filled:filled+k] = prop1[idx]
            y2[filled:filled+k] = prop2[idx]
            filled += k

        trials += batch

    return y1, y2



def simulate_genes(
    n_cells: int,
    pos_params,   # list of (r1, theta1, r2, theta2)
    neg_params,   # list of (r1, theta1, r2, theta2)
    rand_params,  # list of (r, theta)
    lambda_pos: float,
    lambda_neg: float,
    rng=None
):
    """
    Generate a gene expression matrix X with shape (n_cells, n_genes).

    - For each tuple (r1, th1, r2, th2) in pos_params:
    generate a positively correlated gene pair using lambda_pos.
    - For each tuple (r1, th1, r2, th2) in neg_params:
    generate a negatively correlated gene pair using lambda_neg.
    - For each tuple (r, th) in rand_params:
    generate an independent (uncorrelated) NB gene.

    Returns
    -------
    X : np.ndarray
        Shape (n_cells, 2*len(pos_params) + 2*len(neg_params) + len(rand_params)).
    gene_info : list
        Metadata for each generated pair/gene, including type and parameters.
    """

    rng = np.random.default_rng() if rng is None else rng

    genes = []
    gene_info = []

    # positive correlated pairs
    for k, (r1, th1, r2, th2) in enumerate(pos_params):
        y1, y2 = bivariate_nb_sarmanov_r_theta(
            n_cells, r1, th1, r2, th2, lambda_pos, rng=rng
        )
        genes.append(y1)
        genes.append(y2)
        gene_info.append(("pos_pair", k, (r1, th1, r2, th2, lambda_pos)))

    # negative correlated pairs
    for k, (r1, th1, r2, th2) in enumerate(neg_params):
        y1, y2 = bivariate_nb_sarmanov_r_theta(
            n_cells, r1, th1, r2, th2, lambda_neg, rng=rng
        )
        genes.append(y1)
        genes.append(y2)
        gene_info.append(("neg_pair", k, (r1, th1, r2, th2, lambda_neg)))

    # dependent genes
    # for k, (r, th) in enumerate(rand_params):
    #     y = sample_nb_r_theta(r, th, size=n_cells, rng=rng)
    #     genes.append(y)
    #     gene_info.append(("random", k, (r, th)))
    for k, (r1, th1, r2, th2) in enumerate(rand_params):
        y1, y2 = bivariate_nb_sarmanov_r_theta(
            n_cells, r1, th1, r2, th2, 0, rng=rng
        )
        genes.append(y1)
        genes.append(y2)
        gene_info.append(("random_pair", k, (r1, th1, r2, th2, lambda_neg)))

    X = np.stack(genes, axis=1) if len(genes) > 0 else np.empty((n_cells, 0), dtype=int)
    return X, gene_info
