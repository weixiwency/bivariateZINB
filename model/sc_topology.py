import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
import scanpy as sc
import anndata
from tqdm import tqdm
import warnings

# 忽略一些计算中的警告（如log(0)等，会在代码中处理）
warnings.filterwarnings('ignore')


class ZINB_Marginal_Fitter:
    """
    单基因 ZINB 拟合
    获取 NB 部分的 mu, m (dispersion) 以及每个细胞的后验活跃概率 w_i
    """

    def __init__(self, counts):
        self.Y = np.array(counts, dtype=np.float64)
        self.n = len(self.Y)

    def _zinb_nll(self, params):
        """ZINB 负对数似然函数"""
        mu, m, pi = params

        # 约束条件处理
        if mu <= 0 or m <= 0 or not (0 <= pi <= 1):
            return np.inf

        # NB log-probability
        # P_NB(y) = Gamma(y + 1/m) / (Gamma(y+1)Gamma(1/m)) * (1 / (1+m*mu))^(1/m) * ( (m*mu) / (1+m*mu) )^y
        # log P_NB(y)
        coeff = gammaln(self.Y + 1 / m) - gammaln(self.Y + 1) - gammaln(1 / m)
        log_p_nb = coeff - (1 / m) * np.log(1 + m * mu) + self.Y * np.log(m * mu / (1 + m * mu))

        # ZINB likelihood
        # y = 0: P = pi + (1-pi) * P_NB(0)
        # y > 0: P = (1-pi) * P_NB(y)

        # Case y > 0
        mask_pos = self.Y > 0
        log_L_pos = np.log(1 - pi + 1e-10) + log_p_nb[mask_pos]

        # Case y = 0
        mask_zero = self.Y == 0
        p_nb_0 = (1 + m * mu) ** (-1 / m)
        log_L_zero = np.log(pi + (1 - pi) * p_nb_0 + 1e-10)

        return - (np.sum(log_L_pos) + np.sum(log_L_zero))

    def fit(self):
        # 初始化参数猜测 (Moment Estimation)
        mean = np.mean(self.Y)
        var = np.var(self.Y)

        # 简单的 Moment estimator
        if var > mean:
            m_init = (var - mean) / (mean ** 2)
        else:
            m_init = 1e-4  # Underdispersed or Poisson-like

        pi_init = np.sum(self.Y == 0) / self.n
        mu_init = mean / (1 - pi_init + 1e-6)  # 粗略修正均值

        initial_guess = [mu_init, m_init, pi_init]
        bounds = [(1e-5, None), (1e-5, None), (1e-6, 1 - 1e-6)]

        # 优化
        res = minimize(self._zinb_nll, initial_guess, bounds=bounds, method='L-BFGS-B')

        self.mu, self.m, self.pi = res.x
        return self.mu, self.m, self.pi

    def calculate_posterior_active(self):
        """
        计算 w_i = P(Active | Y_i) (即属于 NB 分布而非 structural zero 的概率)
        用于后续 Soft Assignment
        """
        # P_NB(y)
        log_p_nb = (gammaln(self.Y + 1 / self.m) - gammaln(self.Y + 1) - gammaln(1 / self.m)
                    - (1 / self.m) * np.log(1 + self.m * self.mu)
                    + self.Y * np.log(self.m * self.mu / (1 + self.m * self.mu)))
        p_nb = np.exp(log_p_nb)

        # Posterior
        # w_i = ( (1-pi) * P_NB ) / ( pi*delta_0 + (1-pi)*P_NB )

        numerator = (1 - self.pi) * p_nb

        denominator = np.zeros_like(numerator)
        mask_zero = self.Y == 0
        denominator[mask_zero] = self.pi + (1 - self.pi) * p_nb[mask_zero]
        denominator[~mask_zero] = (1 - self.pi) * p_nb[~mask_zero]

        w = numerator / (denominator + 1e-100)
        return w


class Famoye_BNB_Estimator:
    """
    两两基因 Famoye BNB Lambda 估计
    使用 Soft-Weighted Likelihood (只关注活跃状态下的相关性)
    """

    def __init__(self, y1, y2, mu1, m1, mu2, m2, weights):
        self.y1 = y1
        self.y2 = y2
        self.mu1 = mu1
        self.m1 = m1
        self.mu2 = mu2
        self.m2 = m2
        self.weights = weights  # w1 * w2

        # 预计算 c1, c2 (Famoye eq 2.2)
        # c = E[e^(-Y)] = (1 + m*mu*(1 - e^-1))^(-1/m)
        e_inv = np.exp(-1)
        self.c1 = (1 + self.m1 * self.mu1 * (1 - e_inv)) ** (-1 / self.m1)
        self.c2 = (1 + self.m2 * self.mu2 * (1 - e_inv)) ** (-1 / self.m2)

        # 预计算 Famoye 修正项 A_i
        # Term = (e^(-y1) - c1) * (e^(-y2) - c2)
        self.A = (np.exp(-self.y1) - self.c1) * (np.exp(-self.y2) - self.c2)

        # 计算 Lambda 的理论边界，确保概率非负
        # 1 + lambda * A >= 0 对于所有可能的 y1, y2
        # Max A 发生在 y1=0, y2=0
        A_max = (1 - self.c1) * (1 - self.c2)
        # Min A 发生在 y1=0, y2=inf (或反之)
        A_min = min((1 - self.c1) * (0 - self.c2), (0 - self.c1) * (1 - self.c2))

        # Bounds: lambda >= -1/A_max (如果 A_max > 0)
        #         lambda <= -1/A_min (如果 A_min < 0)
        self.lambda_min = -1.0 / A_max if A_max > 0 else -100
        self.lambda_max = -1.0 / A_min if A_min < 0 else 100

        # 稍微收缩边界防止 log(0)
        self.lambda_min += 1e-4
        self.lambda_max -= 1e-4

    def _neg_log_likelihood(self, lam):
        # LogL = sum [ w_i * log(1 + lambda * A_i) ]
        # 注意：这里我们只优化 Famoye 的修正项部分，NB 部分在 Marginal 阶段已定

        term = 1 + lam * self.A
        if np.any(term <= 0):
            return np.inf

        log_l = np.sum(self.weights * np.log(term))
        return -log_l

    def fit(self):
        # 初始猜测 0 (无相关)
        res = minimize(self._neg_log_likelihood, x0=[0.0],
                       bounds=[(self.lambda_min, self.lambda_max)],
                       method='L-BFGS-B')
        return res.x[0]


def analyze_h5ad_topology(h5ad_path, gene_list=None):
    """
    主流程
    """
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    # 如果没指定基因，选高变基因
    if gene_list is None:
        print("Selecting highly variable genes...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=100)  # Demo用100个，实际建议2000
        genes_to_use = adata.var_names[adata.var['highly_variable']]
        # 恢复原始计数用于拟合
        raw_counts = adata[:, genes_to_use].X
        if scipy.sparse.issparse(raw_counts):
            raw_counts = raw_counts.toarray()
    else:
        genes_to_use = gene_list
        raw_counts = adata[:, genes_to_use].X
        if scipy.sparse.issparse(raw_counts):
            raw_counts = raw_counts.toarray()

    n_genes = len(genes_to_use)
    print(f"Analyzing {n_genes} genes...")

    # Store Marginal Params
    marginal_params = {}
    posterior_weights = {}

    print("Phase I: Fitting Marginal ZINB...")
    for i, gene in enumerate(tqdm(genes_to_use)):
        y = raw_counts[:, i]
        fitter = ZINB_Marginal_Fitter(y)
        mu, m, pi = fitter.fit()
        w = fitter.calculate_posterior_active()

        marginal_params[gene] = {'mu': mu, 'm': m, 'pi': pi}
        posterior_weights[gene] = w

    # Store Pairwise Lambda
    # 结果矩阵
    lambda_matrix = pd.DataFrame(index=genes_to_use, columns=genes_to_use)

    print("Phase II: Estimating Pairwise BNB Lambda...")
    # Demo: 只计算部分 pair，实际需并行化
    pairs = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            pairs.append((genes_to_use[i], genes_to_use[j]))

    for g1, g2 in tqdm(pairs):
        y1 = raw_counts[:, list(genes_to_use).index(g1)]
        y2 = raw_counts[:, list(genes_to_use).index(g2)]

        p1 = marginal_params[g1]
        p2 = marginal_params[g2]

        # 联合权重：两个基因都是 Active 的概率
        w_joint = posterior_weights[g1] * posterior_weights[g2]

        # 如果 joint active 的细胞太少，跳过
        if np.sum(w_joint) < 10:
            lambda_matrix.loc[g1, g2] = np.nan
            lambda_matrix.loc[g2, g1] = np.nan
            continue

        famoye = Famoye_BNB_Estimator(y1, y2,
                                      p1['mu'], p1['m'],
                                      p2['mu'], p2['m'],
                                      w_joint)
        lam = famoye.fit()

        lambda_matrix.loc[g1, g2] = lam
        lambda_matrix.loc[g2, g1] = lam

    return marginal_params, lambda_matrix


import scipy.sparse
