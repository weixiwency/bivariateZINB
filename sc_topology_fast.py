import numpy as np
import pandas as pd
from scipy.special import gammaln
import scanpy as sc
import scipy.sparse
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')


class ZINB_Marginal_Fitter:
    """
    单基因 ZINB 拟合 (使用MoM 闭式解)
    """

    def __init__(self, counts):
        self.Y = np.array(counts, dtype=np.float64)
        self.n = len(self.Y)

    def fit_mom(self):
        # 1. 基础统计量
        sample_mean = np.mean(self.Y)
        sample_var = np.var(self.Y)
        prop_zeros = np.sum(self.Y == 0) / self.n

        # 边界保护
        if prop_zeros >= 0.99 or sample_mean < 1e-4:
            return 1e-3, 1e-3, 0.99

        # 2. 估计 Pi (零膨胀率)
        pi_hat = prop_zeros

        # 3. 估计 NB 部分的 Mu
        # E[Y] = (1-pi) * mu
        mu_nb = sample_mean / (1 - pi_hat + 1e-6)

        # 4. 估计 NB 部分的方差
        # Var(Y) = (1-pi)Var_nb + pi(1-pi)mu^2
        inflation_var = pi_hat * (1 - pi_hat) * (mu_nb ** 2)
        var_nb = (sample_var - inflation_var) / (1 - pi_hat + 1e-6)

        # 5. 估计 m
        # Var_nb = mu + m * mu^2
        if var_nb > mu_nb:
            m_hat = (var_nb - mu_nb) / (mu_nb ** 2 + 1e-6)
        else:
            m_hat = 1e-4  # Underdispersion fallback

        self.mu, self.m, self.pi = mu_nb, m_hat, pi_hat
        return self.mu, self.m, self.pi

    def calculate_posterior_active(self):
        """计算 w_i = P(Active | y_i)"""
        # 防止数值溢出
        mu = np.clip(self.mu, 1e-6, 1e4)
        m = np.clip(self.m, 1e-6, 1e4)
        pi = np.clip(self.pi, 0, 1 - 1e-6)

        # 计算 NB 概率 (Log domain)
        r = 1.0 / m
        p = 1.0 / (1.0 + m * mu)

        log_prob_nb = (gammaln(self.Y + r) - gammaln(self.Y + 1) - gammaln(r)
                       + r * np.log(p) + self.Y * np.log(1 - p + 1e-10))

        prob_nb = np.exp(log_prob_nb)

        # 贝叶斯后验公式
        numerator = (1 - pi) * prob_nb
        denominator = np.zeros_like(numerator)

        mask_zero = self.Y == 0
        # P(Y=0) = pi + (1-pi)P_NB(0)
        p_nb_0 = (1 + m * mu) ** (-r)
        denominator[mask_zero] = pi + (1 - pi) * p_nb_0
        denominator[~mask_zero] = (1 - pi) * prob_nb[~mask_zero]

        w = numerator / (denominator + 1e-100)
        return w


class Famoye_BNB_Formula_Estimator:
    """
    Famoye BNB Lambda 闭式解估计
    """

    def __init__(self, y1, y2, p1, p2, w1, w2):
        self.y1 = y1
        self.y2 = y2
        # p1, p2 是字典 {'mu': ..., 'm': ...}
        self.p1 = p1
        self.p2 = p2
        # 联合权重：用于计算加权协方差，剔除 Zero-Inflation 的干扰
        self.w_joint = w1 * w2

    def _calculate_famoye_terms(self, mu, m):
        """
        计算辅助变量 c 和 A
        Mapping: Famoye's theta = m*mu / (1 + m*mu)
        """
        # 1. 映射到 Famoye 的参数 theta
        theta = (m * mu) / (1 + m * mu)

        e_inv = np.exp(-1)

        # 2. 计算 c
        # c = [(1-theta) / (1 - theta*e^-1)] ^ (1/m)
        num = 1 - theta
        den = 1 - theta * e_inv
        c = (num / den) ** (1 / m)

        # 3. 计算 A
        # A = [m^-1 * theta * e^-1 / (1 - theta*e^-1)] - [m^-1 * theta / (1 - theta)]
        #   = (1/m) * [ (theta*e^-1)/(1-theta*e^-1) - theta/(1-theta) ]
        term1 = (theta * e_inv) / (1 - theta * e_inv)
        term2 = theta / (1 - theta)
        A = (1 / m) * (term1 - term2)

        return c, A

    def fit(self):
        """
        Lambda = Cov(Y1, Y2) / (c1 * c2 * A1 * A2)
        """
        # 1. 计算 c1, c2, A1, A2
        c1, A1 = self._calculate_famoye_terms(self.p1['mu'], self.p1['m'])
        c2, A2 = self._calculate_famoye_terms(self.p2['mu'], self.p2['m'])

        # 如果 A 为 0 (意味着没有相关性基础)，直接返回 0
        if abs(A1) < 1e-9 or abs(A2) < 1e-9:
            return 0.0

        # 2. 计算观测数据的协方差
        # 只关心 "NB 成分" 之间的协方差，所以使用 w_joint 加权

        # 加权均值
        sum_w = np.sum(self.w_joint)
        if sum_w < 1e-6: return 0.0  # 权重太小，无法计算

        mean1 = np.sum(self.y1 * self.w_joint) / sum_w
        mean2 = np.sum(self.y2 * self.w_joint) / sum_w

        # 加权协方差公式: sum(w * (y1-m1)*(y2-m2)) / (sum_w - 1)
        cov_12 = np.sum(self.w_joint * (self.y1 - mean1) * (self.y2 - mean2)) / (sum_w - 1 + 1e-6)

        # 3. 论文闭式解
        # Lambda = Cov / (c1 * c2 * A1 * A2)
        denominator = c1 * c2 * A1 * A2
        lam = cov_12 / (denominator + 1e-12)

        return lam


def analyze_h5ad_topology_fast(h5ad_path, gene_list=None):
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    # 确保使用 Raw Counts
    if adata.raw is not None:
        X = adata.raw[:, adata.var_names].X
    else:
        X = adata.X

    if scipy.sparse.issparse(X):
        X = X.toarray()

    if gene_list is None:
        # 默认选前 50 个高变基因演示
        print("Selecting top 50 genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=50)
        gene_indices = np.where(adata.var['highly_variable'])[0][:50]
        genes_to_use = adata.var_names[gene_indices]
        counts_subset = X[:, gene_indices]
    else:
        # 找到对应基因的索引
        genes_to_use = gene_list
        indices = [list(adata.var_names).index(g) for g in gene_list]
        counts_subset = X[:, indices]

    n_genes = len(genes_to_use)

    # --- Phase I: Marginal Parameters (矩估计) ---
    print(f"Phase I: Estimating Marginal Parameters for {n_genes} genes...")
    marginal_params = {}
    posterior_weights = {}  # 存储 w 用于第二步

    for i, gene in enumerate(genes_to_use):
        y = counts_subset[:, i]
        fitter = ZINB_Marginal_Fitter(y)
        mu, m, pi = fitter.fit_mom()  # 使用极速矩估计
        w = fitter.calculate_posterior_active()

        marginal_params[gene] = {'mu': mu, 'm': m, 'pi': pi}
        posterior_weights[gene] = w

    # --- Phase II: Pairwise Lambda (闭式解) ---
    print("Phase II: Calculating Pairwise Lambda (Closed-form)...")
    lambda_matrix = pd.DataFrame(0.0, index=genes_to_use, columns=genes_to_use)

    # 遍历所有对 (i, j)
    count = 0
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            g1, g2 = genes_to_use[i], genes_to_use[j]
            y1, y2 = counts_subset[:, i], counts_subset[:, j]

            w1 = posterior_weights[g1]
            w2 = posterior_weights[g2]

            estimator = Famoye_BNB_Formula_Estimator(y1, y2,
                                                     marginal_params[g1],
                                                     marginal_params[g2],
                                                     w1, w2)
            lam = estimator.fit()

            lambda_matrix.loc[g1, g2] = lam
            lambda_matrix.loc[g2, g1] = lam
            count += 1

    return marginal_params, lambda_matrix
