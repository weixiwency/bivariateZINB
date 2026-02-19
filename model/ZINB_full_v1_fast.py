import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln
import itertools
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 模块 1: BZINB 底层模型 
# =====================================================================

class Smart_Initializer:
    @staticmethod
    def init_pis_by_counts(y1, y2):
        N = len(y1)
        n1 = np.sum((y1 > 0) & (y2 > 0))  
        n2 = np.sum((y1 == 0) & (y2 > 0)) 
        n3 = np.sum((y1 > 0) & (y2 == 0)) 
        n4 = np.sum((y1 == 0) & (y2 == 0))  
        counts = np.array([n1, n2, n3, n4]) + max(1.0, N*0.01)
        return counts / np.sum(counts)

    @staticmethod
    def get_marginal_zinb_params(y):
        n = len(y)
        mean_val = np.mean(y)
        var_val = np.var(y)
        p0 = np.sum(y == 0) / n
        pi_hat = min(p0, 0.95)
        mu_nb = mean_val / (1 - pi_hat + 1e-6)
        inflation_term = pi_hat * (1 - pi_hat) * (mu_nb ** 2)
        var_nb = (var_val - inflation_term) / (1 - pi_hat + 1e-6)
        var_nb = max(var_nb , mu_nb + 1e-4)
        m_init = (var_nb - mu_nb) / (mu_nb ** 2 + 1e-6) if var_nb > mu_nb else 1.0
        theta_init = (m_init * mu_nb) / (1 + m_init * mu_nb)
        return max(m_init, 0.1), max(theta_init, 0.05)


class BZINB_Model:
    def __init__(self, tol=1e-3, max_iter=20, lambda_reg=0.2): 
        self.tol = tol
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.opt_options = {'maxiter': 3, 'xatol': 1e-1, 'fatol': 1e-1} 

    def nb_logpmf(self, y, m, theta):
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        m = np.clip(m, 1e-6, 1e4)
        r = 1.0 / m
        coeff = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        return coeff + y * np.log(theta) + r * np.log(1 - theta)

    def get_famoye_term(self, y, m, t):
        e_inv = np.exp(-1)
        base = (1 - t) / (1 - t * e_inv)
        c = base ** (1 / m)
        return np.exp(-y) - c

    def bnb_logpmf(self, y1, y2, m1, t1, m2, t2, lam):
        log_p1 = self.nb_logpmf(y1, m1, t1)
        log_p2 = self.nb_logpmf(y2, m2, t2)
        term1 = self.get_famoye_term(y1, m1, t1)
        term2 = self.get_famoye_term(y2, m2, t2)
        correction = np.maximum(1 + lam * term1 * term2, 1e-10)
        return log_p1 + log_p2 + np.log(correction)

    def fit(self, y1, y2, constraint=None, init_params=None):
        pis = Smart_Initializer.init_pis_by_counts(y1, y2)
        
        if constraint == 'p2_0': pis[1] = 0.0
        elif constraint == 'p3_0': pis[2] = 0.0
        elif constraint == 'p2p3_0': pis[1] = 0.0; pis[2] = 0.0
        elif constraint == 'p1_0': pis[0] = 0.0
        pis = pis / (np.sum(pis) + 1e-10) 

        if init_params is not None:
            params = init_params.copy()
        else:
            m1_in, t1_in = Smart_Initializer.get_marginal_zinb_params(y1)
            m2_in, t2_in = Smart_Initializer.get_marginal_zinb_params(y2)
            params = {'m1': m1_in, 't1': t1_in, 'm2': m2_in, 't2': t2_in, 'lam': 0.0}
        
        log_lik_history = []
        curr_ll = -np.inf

        for iteration in range(self.max_iter):
            lp1 = self.bnb_logpmf(y1, y2, params['m1'], params['t1'], params['m2'], params['t2'], params['lam'])
            p1 = np.exp(lp1) * pis[0]
            p2 = np.exp(self.nb_logpmf(y2, params['m2'], params['t2'])) * (y1 == 0) * pis[1]
            p3 = np.exp(self.nb_logpmf(y1, params['m1'], params['t1'])) * (y2 == 0) * pis[2]
            p4 = ((y1 == 0) & (y2 == 0)).astype(float) * pis[3]

            total_prob = p1 + p2 + p3 + p4 + 1e-20
            gamma = np.vstack([p1, p2, p3, p4]) / total_prob
            gamma = gamma.T

            pis = gamma.mean(axis=0)

            w1 = gamma[:, 0] + gamma[:, 2] 
            w_corr = gamma[:, 0]
            term2_fixed = self.get_famoye_term(y2, params['m2'], params['t2'])

            def loss_g1(p):
                ll_nb = np.sum(w1 * self.nb_logpmf(y1, p[0], p[1]))
                corr = np.maximum(1 + params['lam'] * self.get_famoye_term(y1, p[0], p[1]) * term2_fixed, 1e-10)
                return -(ll_nb + np.sum(w_corr * np.log(corr)))

            res1 = minimize(loss_g1, [params['m1'], params['t1']], bounds=[(0.01, 20), (0.01, 0.999)], 
                            method='Nelder-Mead', options=self.opt_options)
            params['m1'], params['t1'] = res1.x

            w2 = gamma[:, 0] + gamma[:, 1] 
            term1_fixed = self.get_famoye_term(y1, params['m1'], params['t1'])

            def loss_g2(p):
                ll_nb = np.sum(w2 * self.nb_logpmf(y2, p[0], p[1]))
                corr = np.maximum(1 + params['lam'] * term1_fixed * self.get_famoye_term(y2, p[0], p[1]), 1e-10)
                return -(ll_nb + np.sum(w_corr * np.log(corr)))

            res2 = minimize(loss_g2, [params['m2'], params['t2']], bounds=[(0.01, 20), (0.01, 0.999)], 
                            method='Nelder-Mead', options=self.opt_options)
            params['m2'], params['t2'] = res2.x

            t1_f, t2_f = self.get_famoye_term(y1, params['m1'], params['t1']), self.get_famoye_term(y2, params['m2'], params['t2'])
            def loss_lam(l):
                corr = np.maximum(1 + l[0] * t1_f * t2_f, 1e-10)
                return -np.sum(gamma[:, 0] * np.log(corr)) + self.lambda_reg * (l[0]**2)

            res_lam = minimize(loss_lam, [params['lam']], bounds=[(-10, 10)], 
                               method='Nelder-Mead', options=self.opt_options)
            params['lam'] = res_lam.x[0]

            curr_ll = np.sum(np.log(total_prob))
            log_lik_history.append(curr_ll)
            if iteration > 0 and abs(log_lik_history[-1] - log_lik_history[-2]) < self.tol:
                break

        return params, pis, curr_ll

# =====================================================================
# 模块 2: 推断函数与 Worker 进程封装
# =====================================================================

def step1_univariate_filter(X, gene_names, corr_threshold=0.8):
    n_genes = X.shape[1]
    exog = np.ones((X.shape[0], 1))
    passed_genes_indices = []
    
    print(f"--- 步骤 1: 独立基因严格筛选 (Covariance Corr < {corr_threshold}) ---")
    for i in tqdm(range(n_genes), desc="Univariate Filter"):
        y = X[:, i]
        if np.mean(y) < 0.05: continue 
        try:
            model = ZeroInflatedNegativeBinomialP(y, exog, exog, inflation='logit', p=2)
            res = model.fit(method='nm', maxiter=200, disp=0)
            if not res.mle_retvals['converged']: continue
                
            cov_matrix = res.cov_params()
            std_devs = np.sqrt(np.diag(cov_matrix))
            if np.any(std_devs == 0): continue 
            
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            corr_pi_mu = abs(corr_matrix.iloc[0, 1] if hasattr(corr_matrix, 'iloc') else corr_matrix[0, 1])
            corr_pi_alpha = abs(corr_matrix.iloc[0, 2] if hasattr(corr_matrix, 'iloc') else corr_matrix[0, 2])
            
            if corr_pi_mu < corr_threshold and corr_pi_alpha < corr_threshold:
                passed_genes_indices.append(i)
        except:
            continue
            
    return passed_genes_indices

def lrt_pvalue(ll_full, ll_constrained, df):
    D = max(2 * (ll_full - ll_constrained), 0)
    return 0.5 * stats.chi2.sf(D, df) 

def step3_bivariate_decision(yA, yB, marginal_A, marginal_B, alpha=0.05):
    model_full = BZINB_Model(max_iter=20)
    model_restr = BZINB_Model(max_iter=10) 
    
    try:
        init_p = {'m1': marginal_A[0], 't1': marginal_A[1], 'm2': marginal_B[0], 't2': marginal_B[1], 'lam': 0.0}
        
        params_f, _, ll_full = model_full.fit(yA, yB, constraint=None, init_params=init_p)
        _, _, ll_p2_0 = model_restr.fit(yA, yB, constraint='p2_0', init_params=params_f)
        _, _, ll_p3_0 = model_restr.fit(yA, yB, constraint='p3_0', init_params=params_f)
        _, _, ll_p2p3_0 = model_restr.fit(yA, yB, constraint='p2p3_0', init_params=params_f)
        _, _, ll_p1_0 = model_restr.fit(yA, yB, constraint='p1_0', init_params=params_f)
        
        accept_p2 = lrt_pvalue(ll_full, ll_p2_0, df=1) > alpha
        accept_p3 = lrt_pvalue(ll_full, ll_p3_0, df=1) > alpha
        
        if accept_p2 and accept_p3:
            if lrt_pvalue(ll_full, ll_p2p3_0, df=2) > alpha: return "Binary Co-expression (共表达)"
        if accept_p2 and not accept_p3: return "A Contains B (A包含B)"
        if accept_p3 and not accept_p2: return "B Contains A (B包含A)"
        if lrt_pvalue(ll_full, ll_p1_0, df=1) > alpha: return "Mutual Exclusivity (互斥)"
            
        lam_val = params_f['lam']
        if abs(lam_val) > 0.1: 
            if lam_val > 0: return "Continuous Synergistic (连续协同)"
            else: return "Continuous Antagonistic (连续拮抗/矛盾关系)"
                
        return "Independent (独立无关)"
    except Exception:
        return "Independent (独立无关)" 

# 【新增】：多进程 Worker 函数
def process_single_pair(task_tuple, X, marginals_dict, gene_names):
    """
    处理单个基因对的任务函数。
    task_tuple 包含 (i, j, idx_A, idx_B, is_valid)
    """
    i, j, idx_A, idx_B, is_valid = task_tuple
    name_A = gene_names[idx_A]
    name_B = gene_names[idx_B]
    
    # 查表未通过的，直接返回
    if not is_valid:
        return {'Gene_A': name_A, 'Gene_B': name_B, 'Relationship': "Independent (独立无关)"}
        
    yA = X[:, idx_A]
    yB = X[:, idx_B]
    
    relation = step3_bivariate_decision(
        yA, yB, 
        marginal_A=marginals_dict[idx_A], 
        marginal_B=marginals_dict[idx_B], 
        alpha=0.05
    )
    return {'Gene_A': name_A, 'Gene_B': name_B, 'Relationship': relation}

def build_three_matrices(results, passed_genes):
    mat_binary = pd.DataFrame(0, index=passed_genes, columns=passed_genes)
    mat_directed = pd.DataFrame(0, index=passed_genes, columns=passed_genes)
    mat_continuous = pd.DataFrame(0, index=passed_genes, columns=passed_genes)
    
    for row in results:
        g_A, g_B, rel = row['Gene_A'], row['Gene_B'], row['Relationship']
        if rel == "Binary Co-expression (共表达)":
            mat_binary.loc[g_A, g_B] = 1; mat_binary.loc[g_B, g_A] = 1
        elif rel == "Mutual Exclusivity (互斥)":
            mat_binary.loc[g_A, g_B] = -1; mat_binary.loc[g_B, g_A] = -1
        elif rel == "A Contains B (A包含B)":
            mat_directed.loc[g_A, g_B] = 1 
        elif rel == "B Contains A (B包含A)":
            mat_directed.loc[g_B, g_A] = 1
        elif rel == "Continuous Synergistic (连续协同)":
            mat_continuous.loc[g_A, g_B] = 1; mat_continuous.loc[g_B, g_A] = 1
        elif rel == "Continuous Antagonistic (连续拮抗/矛盾关系)":
            mat_continuous.loc[g_A, g_B] = -1; mat_continuous.loc[g_B, g_A] = -1

    return mat_binary, mat_directed, mat_continuous

# =====================================================================
# 主流程 (多核并行版)
# =====================================================================

def main():
    print("加载数据 adata.h5ad ...")
    try:
        adata = sc.read_h5ad(r'/home/weixi/Desktop/bivariateZINB/adata.h5ad')
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        gene_names = np.array(adata.var_names)
    except FileNotFoundError:
        print("未找到 adata.h5ad 文件。")
        return

    # 1. 单基因筛选
    passed_indices = step1_univariate_filter(X, gene_names, corr_threshold=0.8)
    passed_genes = gene_names[passed_indices]
    print(f"筛选出 {len(passed_genes)} 个符合条件的基因。")
    if len(passed_genes) < 2: return
        
    # 2. 缓存边缘分布
    print("\n--- 预计算单基因边缘分布参数 (Caching) ---")
    marginals_dict = {}
    for idx in tqdm(passed_indices, desc="Marginals Cache"):
        marginals_dict[idx] = Smart_Initializer.get_marginal_zinb_params(X[:, idx])
        
    # 3. 极速矩阵粗筛
    print("\n--- 预计算全局查表矩阵 (极速粗筛) ---")
    X_passed = X[:, passed_indices] 
    corr_matrix, _ = stats.spearmanr(X_passed)
    corr_matrix = np.nan_to_num(corr_matrix) 
    
    X_bin = (X_passed > 0).astype(int)
    intersection = X_bin.T.dot(X_bin) 
    sizes = X_bin.sum(axis=0)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard_matrix = intersection / np.clip(union, 1, None)
    
    # 4. 准备并行任务列队
    gene_pairs = list(itertools.combinations(range(len(passed_indices)), 2))
    tasks = []
    
    # 提前用矩阵查表标记哪些是对子需要跑 BZINB (is_valid = True)
    valid_count = 0
    for i, j in gene_pairs:
        idx_A = passed_indices[i]
        idx_B = passed_indices[j]
        
        jac_val = jaccard_matrix[i, j]
        corr_val = abs(corr_matrix[i, j])
        
        # 宽容度阈值，通过的才进行复杂运算
        is_valid = not (jac_val < 0.05 and corr_val < 0.1)
        if is_valid: valid_count += 1
            
        tasks.append((i, j, idx_A, idx_B, is_valid))
        
    print(f"\n--- 开始并行推断 {len(tasks)} 对基因网络 (其中 {valid_count} 对进入深度计算) ---")
    
    # 【核心修改】：获取服务器核心数并启动进程池
    # 自动探测 CPU 核心，保留一个核给系统以免服务器卡死
    n_cores = max(1, mp.cpu_count() - 1)
    print(f"🚀 检测到服务器，已启动 {n_cores} 个 Worker 进程火力全开！")
    
    # 绑定全局参数到 worker 函数
    worker = partial(process_single_pair, X=X, marginals_dict=marginals_dict, gene_names=gene_names)
    
    results = []
    # 启动多进程池
    with mp.Pool(processes=n_cores) as pool:
        # imap_unordered 可以和 tqdm 完美结合显示多核进度条，并且效率最高
        for res in tqdm(pool.imap_unordered(worker, tasks, chunksize=100), total=len(tasks), desc="Parallel Computing"):
            results.append(res)
            
    # 5. 保存结果
    print("\n--- 构建并保存三大邻接矩阵 ---")
    mat_binary, mat_directed, mat_continuous = build_three_matrices(results, passed_genes)
    
    mat_binary.to_csv("Matrix_1_Binary_CoBursting.csv")
    mat_directed.to_csv("Matrix_2_Directional_Inclusion.csv")
    mat_continuous.to_csv("Matrix_3_Continuous_Regulation.csv")
    
    print("分析完成！所有结果已保存。")

if __name__ == "__main__":
    # 在 Linux 服务器上建议使用 forkserver 或 spawn 来避免内存泄漏
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass
    main()