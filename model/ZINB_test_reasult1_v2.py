import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
import itertools
from tqdm import tqdm
import multiprocessing as mp
import os
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

# 导入你的主程序模块
import ZINB_full_v1_fast as zinb 

# ==========================================
# 0. 补充缺失的 rho 计算公式 (独立函数，防报错)
# ==========================================
def safe_compute_rho(m1, t1, m2, t2, lam):
    """
    计算 BZINB 模型的连续部分相关系数 rho
    公式: rho = lam * c1 * c2 * A1 * A2 / (sigma1 * sigma2)
    """
    e_inv = np.exp(-1.0)
    c1 = ((1.0 - t1) / (1.0 - t1 * e_inv)) ** (1.0 / m1)
    c2 = ((1.0 - t2) / (1.0 - t2 * e_inv)) ** (1.0 / m2)
    
    mu1 = t1 / (m1 * (1.0 - t1))
    mu2 = t2 / (m2 * (1.0 - t2))
    
    A1 = mu1 * (e_inv - 1.0) / (1.0 - t1 * e_inv)
    A2 = mu2 * (e_inv - 1.0) / (1.0 - t2 * e_inv)
    
    sigma1 = np.sqrt(mu1 + m1 * (mu1 ** 2))
    sigma2 = np.sqrt(mu2 + m2 * (mu2 ** 2))
    
    cov = lam * c1 * c2 * A1 * A2
    return float(cov / (sigma1 * sigma2))


# ==========================================
# 1. 内存防爆：多进程共享内存初始化
# ==========================================
# 这三个全局变量专门供子进程读取，避免几十万次的数据复制
global_X = None
global_marginals = None
global_genes = None

def init_worker(X_shared, marginals_shared, genes_shared):
    """
    子进程启动时的初始化函数。
    把大矩阵一次性存入内存，后续所有任务直接查内存！
    """
    global global_X, global_marginals, global_genes
    global_X = X_shared
    global_marginals = marginals_shared
    global_genes = genes_shared


# ==========================================
# 2. 自定义推断流程
# ==========================================
def custom_step3_bivariate_decision(yA, yB, marginal_A, marginal_B, alpha=0.02):
    model_full = zinb.BZINB_Model()
    model_restr = zinb.BZINB_Model()

    init_p = {'m1': marginal_A[0], 't1': marginal_A[1],
              'm2': marginal_B[0], 't2': marginal_B[1], 'lam': 0.0}

    # 拟合全模型
    params_f, _, ll_full = model_full.fit(yA, yB, constraint=None, init_params=init_p)
    # 拟合约束模型
    _, _, ll_p2_0 = model_restr.fit(yA, yB, constraint='p2_0', init_params=params_f)
    _, _, ll_p3_0 = model_restr.fit(yA, yB, constraint='p3_0', init_params=params_f)
    _, _, ll_p1_0 = model_restr.fit(yA, yB, constraint='p1_0', init_params=params_f)

    accept_p2 = zinb.lrt_pvalue(ll_full, ll_p2_0, 1) > alpha
    accept_p3 = zinb.lrt_pvalue(ll_full, ll_p3_0, 1) > alpha

    # 使用前面定义的独立公式计算 Rho
    rho_val = safe_compute_rho(params_f['m1'], params_f['t1'], params_f['m2'], params_f['t2'], params_f['lam'])

    relation = "Independent (独立无关)"
    if accept_p2 and accept_p3:
        relation = "Binary Co-expression (共表达)"
    elif accept_p2 ^ accept_p3:
        relation = "A Contains B (A包含B)" if accept_p2 else "B Contains A (B包含A)"
    elif zinb.lrt_pvalue(ll_full, ll_p1_0, 1) > alpha:
        relation = "Mutual Exclusivity (互斥)"
    elif abs(rho_val) > 0.05: # 连续相关性阈值设为 0.05
        relation = "Continuous Synergistic (连续协同)" if rho_val > 0 else "Continuous Antagonistic (连续拮抗)"

    return {
        'Relationship': relation,
        'LL_full': ll_full,
        'Rho': rho_val,
        'm1': params_f['m1'], 't1': params_f['t1'],
        'm2': params_f['m2'], 't2': params_f['t2'],
        'lam': params_f['lam']
    }

def custom_worker(task_tuple):
    """
    工作进程：不再接收整个大矩阵，只接收当前任务的坐标 (task_tuple)。
    需要数据时直接从全局变量中提取。
    """
    i, j, idx_A, idx_B, is_valid = task_tuple
    name_A, name_B = global_genes[idx_A], global_genes[idx_B]
    
    if not is_valid:
        return {'Gene_A': name_A, 'Gene_B': name_B, 'Relationship': "Independent (独立无关)", 
                'LL_full': np.nan, 'Rho': np.nan, 'm1': np.nan, 't1': np.nan, 'm2': np.nan, 't2': np.nan, 'lam': np.nan}
        
    # 从共享内存获取数据
    yA, yB = global_X[:, idx_A], global_X[:, idx_B]
    res_dict = custom_step3_bivariate_decision(yA, yB, global_marginals[idx_A], global_marginals[idx_B], alpha=0.02)
    res_dict.update({'Gene_A': name_A, 'Gene_B': name_B})
    return res_dict


# ==========================================
# 3. 梳理基因模块 (图网络分析)
# ==========================================
def analyze_gene_modules(df_results):
    print("\n--- 开始划分基因模块 ---")
    df_sig = df_results[df_results['Relationship'] != "Independent (独立无关)"]
    coexp_edges = df_sig[df_sig['Relationship'] == "Binary Co-expression (共表达)"]
    G_coexp = nx.Graph()
    for _, row in coexp_edges.iterrows():
        weight_type = "Positive" if row['Rho'] > 0 else "Negative"
        G_coexp.add_edge(row['Gene_A'], row['Gene_B'], weight=row['Rho'], type=weight_type)
    
    modules = list(nx.connected_components(G_coexp))
    modules = [list(m) for m in modules if len(m) > 2] 
    
    print(f"发现 {len(modules)} 个包含 3 个以上基因的共表达模块。")
    with open("gene_modules_report.txt", "w") as f:
        f.write(f"共发现 {len(modules)} 个主要共表达模块:\n\n")
        for idx, mod in enumerate(modules):
            f.write(f"Module {idx+1} (大小 {len(mod)}): {', '.join(mod[:10])}{'...' if len(mod)>10 else ''}\n")
            subgraph = G_coexp.subgraph(mod)
            pos_edges = sum(1 for u, v, d in subgraph.edges(data=True) if d['type'] == 'Positive')
            neg_edges = sum(1 for u, v, d in subgraph.edges(data=True) if d['type'] == 'Negative')
            f.write(f"  -> 内部正相关边数: {pos_edges}, 负相关边数: {neg_edges}\n\n")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    file_path =r"/home/weixi/Desktop/Omics/chengwang_data/ICML_data/real_data/preprocessed_t-cell-depleted-bm-rna.h5ad"
    print(f"正在加载真实数据集: {file_path} ...")
    if not os.path.exists(file_path):
        print("找不到文件路径，请检查路径。")
        return

    adata = sc.read_h5ad(file_path)
    
    # 取 HVG，限制 1000 个基因
    adata = adata[:, :1000].copy()

    adata.X = adata.X.toarray()
    
    adata.X = np.round(np.expm1(adata.X))
    

    all_genes = np.array(adata.var_names)
    X = adata.X

    print("\n--- 步骤 1: 单基因筛选 ---")
    passed_indices = zinb.step1_univariate_filter(X, all_genes, corr_threshold=0.8)
    passed_genes = all_genes[passed_indices]
    
    pd.DataFrame({'Kept_Genes': passed_genes}).to_csv("Debug_Kept_Genes.csv", index=False)
    print(f"保留了 {len(passed_genes)} 个符合模型假设的基因。")
    if len(passed_genes) < 2: return

    print("\n--- 步骤 2: 预计算与矩阵查表 ---")
    marginals_dict = {}
    for idx in tqdm(passed_indices, desc="Marginals Cache"):
        marginals_dict[idx] = zinb.Smart_Initializer.get_marginal_zinb_params(X[:, idx])
        
    X_passed = X[:, passed_indices] 
    corr_matrix, _ = stats.spearmanr(X_passed)
    corr_matrix = np.nan_to_num(corr_matrix) 
    X_bin = (X_passed > 0).astype(int)
    intersection = X_bin.T.dot(X_bin) 
    sizes = X_bin.sum(axis=0)
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard_matrix = intersection / np.clip(union, 1, None)
    
    gene_pairs = list(itertools.combinations(range(len(passed_indices)), 2))
    tasks = []
    
    valid_count = 0
    for i, j in gene_pairs:
        idx_A, idx_B = passed_indices[i], passed_indices[j]
        is_valid = not (jaccard_matrix[i, j] < 0.05 and abs(corr_matrix[i, j]) < 0.1)
        if is_valid: valid_count += 1
        tasks.append((i, j, idx_A, idx_B, is_valid))
        
    # ================= 并行计算 (防爆版) =================
    # 给系统预留充足的资源，避免卡死
    n_cores = max(1, mp.cpu_count() - 4)
    results = []
    print(f"\n--- 开始并行推断 {len(tasks)} 对基因网络 ---")
    print(f"-> 其中 {valid_count} 对进入深度计算")
    print(f"-> 分配 {n_cores} 核心")
    
    # 使用 initializer 把大矩阵传给子进程，避免重复复制！
    with mp.Pool(processes=n_cores, initializer=init_worker, initargs=(X, marginals_dict, all_genes)) as pool:
        # chunksize 调到 500，进一步降低通信频率
        for res in tqdm(pool.imap_unordered(custom_worker, tasks, chunksize=500), total=len(tasks)):
            results.append(res)

    print("\n--- 正在保存结果 ---")
    df_results = pd.DataFrame(results)
    df_results.to_csv("Debug_Full_Detailed_Results.csv", index=False)
    print("-> 参数与似然结果已保存至 Debug_Full_Detailed_Results.csv")

    analyze_gene_modules(df_results)

if __name__ == "__main__":
    try:
        # forkserver 是 Linux 下最安全、最省内存的多进程启动方式
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass
    main()