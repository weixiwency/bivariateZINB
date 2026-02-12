import numpy as np
import pandas as pd
from scipy.special import gammaln
import scanpy as sc
import scipy.sparse
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')


class BNB_Marginal_Estimator:
    """
    单基因 BNB 拟合 (使用MoM 闭式解)
    """

    def __init__(self, counts):
        self.Y = np.array(counts, dtype=np.float64)
        self.n = len(self.Y)

    def fit_mom(self):
    
        sample_mean = np.mean(self.Y)
        sample_var = np.var(self.Y, ddof=1)

        theta = 1.0 - (sample_mean / (sample_var))
        m = theta / (sample_mean * (1.0 - theta))
        r = 1.0 / m

        self.r = r
        self.theta = theta

        return self.r, self.theta

    # def calculate_posterior_active(self):
    #     """计算 w_i = P(Active | y_i)"""
    #     # 防止数值溢出
    #     mu = np.clip(self.mu, 1e-6, 1e4)
    #     m = np.clip(self.m, 1e-6, 1e4)
    #     pi = np.clip(self.pi, 0, 1 - 1e-6)

    #     # 计算 NB 概率 (Log domain)
    #     r = 1.0 / m
    #     p = 1.0 / (1.0 + m * mu)

    #     log_prob_nb = (gammaln(self.Y + r) - gammaln(self.Y + 1) - gammaln(r)
    #                    + r * np.log(p) + self.Y * np.log(1 - p + 1e-10))

    #     prob_nb = np.exp(log_prob_nb)

    #     # 贝叶斯后验公式
    #     numerator = (1 - pi) * prob_nb
    #     denominator = np.zeros_like(numerator)

    #     mask_zero = self.Y == 0
    #     # P(Y=0) = pi + (1-pi)P_NB(0)
    #     p_nb_0 = (1 + m * mu) ** (-r)
    #     denominator[mask_zero] = pi + (1 - pi) * p_nb_0
    #     denominator[~mask_zero] = (1 - pi) * prob_nb[~mask_zero]

    #     w = numerator / (denominator + 1e-100)
    #     return w


class Famoye_BNB_Formula_Estimator:
    """
    Famoye BNB Lambda 闭式解估计
    """

    def __init__(self, y1, y2, r1, theta1, r2, theta2):
        self.y1 = np.asarray(y1, dtype=np.float64)
        self.y2 = np.asarray(y2, dtype=np.float64)
        self.n = self.y1.size
        self.r1, self.theta1 = float(r1), float(theta1)
        self.r2, self.theta2 = float(r2), float(theta2)
        
    @staticmethod
    def _c_A(r, theta):
        e_inv = np.exp(-1.0)
        # c = ((1-theta)/(1-theta e^-1))^r
        c = ((1.0 - theta) / (1.0 - theta * e_inv)) ** r
        # A = r*(theta e^-1/(1-theta e^-1) - theta/(1-theta))
        A = r * ((theta * e_inv) / (1.0 - theta * e_inv) - theta / (1.0 - theta))
        return c, A

    def fit(self, eps=1e-12):
        
        y1bar = float(np.mean(self.y1))
        y2bar = float(np.mean(self.y2))
        s12 = float(np.sum((self.y1 - y1bar) * (self.y2 - y2bar)) / (self.n - 1 + eps))

        c1, A1 = self._c_A(self.r1, self.theta1)
        c2, A2 = self._c_A(self.r2, self.theta2)

        denom = c1 * c2 * A1 * A2
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return 0.0

        lam = s12 / (denom + eps)
        return float(lam)


def analyze_array_topology_fast(
    X,
    gene_names=None,
    pair_mode="all",   # "all" 或 "adjacent_pairs" (0-1,2-3,...)
    eps=1e-12
):
    """
    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_genes)
        count matrix
    gene_names : list[str] or None
        optional names for genes (length n_genes). If None, use "g0","g1",...
    pair_mode : str
        "all" : compute lambda for all pairs (i<j)
        "adjacent_pairs" : only compute (0,1), (2,3), ... pairs
    eps : float
        numerical epsilon

    Returns
    -------
    marginal_params : dict
        {gene: {"r": r, "theta": theta}}
    lambda_matrix : pd.DataFrame
        symmetric matrix of lambda estimates
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_cells, n_genes)")

    n_cells, n_genes = X.shape

    if gene_names is None:
        genes = [f"g{i}" for i in range(n_genes)]
    else:
        if len(gene_names) != n_genes:
            raise ValueError("gene_names length must equal number of genes (X.shape[1])")
        genes = list(gene_names)

    # --- Phase I: per-gene marginal (r, theta) ---
    marginal_params = {}
    r_arr = np.zeros(n_genes, dtype=float)
    theta_arr = np.zeros(n_genes, dtype=float)

    for j, g in enumerate(genes):
        y = X[:, j]
        fitter = BNB_Marginal_Estimator(y)
        r, theta = fitter.fit_mom()
        r_arr[j] = r
        theta_arr[j] = theta
        marginal_params[g] = {"r": float(r), "theta": float(theta)}

    # --- Phase II: pairwise lambda ---
    lambda_matrix = pd.DataFrame(0.0, index=genes, columns=genes)

    if pair_mode == "all":
        pairs = [(i, j) for i in range(n_genes) for j in range(i + 1, n_genes)]
    elif pair_mode == "adjacent_pairs":
        pairs = [(i, i + 1) for i in range(0, n_genes - 1, 2)]
    else:
        raise ValueError('pair_mode must be "all" or "adjacent_pairs"')

    for i, j in pairs:
        y1 = X[:, i]
        y2 = X[:, j]

        est = Famoye_BNB_Formula_Estimator(
            y1, y2,
            r1=r_arr[i], theta1=theta_arr[i],
            r2=r_arr[j], theta2=theta_arr[j]
        )
        lam = est.fit(eps=eps)

        gi, gj = genes[i], genes[j]
        lambda_matrix.loc[gi, gj] = lam
        lambda_matrix.loc[gj, gi] = lam

    return marginal_params, lambda_matrix

