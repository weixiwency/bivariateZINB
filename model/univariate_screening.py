import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import warnings

# 忽略 statsmodels 在迭代过程中的一些警告
warnings.filterwarnings('ignore')


class ZINB_Single_Tester:
    def __init__(self):
        pass

    def fit_and_check(self, y, gene_name="Gene"):
        # 1. 硬过滤 (保持不变)
        if np.sum(y > 0) < 10:
            return {'status': 'Fail', 'reason': 'Strict Filter: Cells < 10'}
        if np.mean(y) < 0.10:
            return {'status': 'Fail', 'reason': 'Strict Filter: Mean < 0.15'}

        # 2. 拟合 (保持不变)
        n_samples = len(y)
        exog = np.ones((n_samples, 1))
        exog_infl = np.ones((n_samples, 1))

        try:
            model = ZeroInflatedNegativeBinomialP(endog=y, exog=exog, exog_infl=exog_infl, inflation='logit', p=2)

            # 自动重试机制 (保持不变)
            fit_methods = ['bfgs', 'nm']
            res = None
            for method in fit_methods:
                try:
                    max_it = 500 if method == 'nm' else 100
                    temp_res = model.fit(method=method, maxiter=max_it, disp=0)
                    if temp_res.mle_retvals['converged']:
                        res = temp_res
                        break
                except:
                    continue

            if res is None: return {'status': 'Fail', 'reason': 'Optimizer not converged'}

            # ==========================================================
            # [核心升级] 使用多变量 Wald Test
            # ==========================================================

            # 提取参数
            params = res.params
            beta_infl = params[0]
            hat_pi = 1.0 / (1.0 + np.exp(-beta_infl))
            hat_alpha = params[2]

            # 1. 稳定性检查 (Condition Number)
            # 这是一个真正的"多变量"体检指标。
            # 检查协方差矩阵的条件数。如果太大，说明参数之间共线性极其严重（模型不可识别）。
            try:
                cov_matrix = res.cov_params()
                cond_num = np.linalg.cond(cov_matrix)
            except:
                cond_num = np.inf

            if cond_num > 1e6:  # 阈值可调，1e6 说明矩阵接近奇异
                return {'status': 'Fail', 'reason': 'Multivariate: Singular Covariance (Unidentifiable)'}

            # 2. 对 Inflation 进行 Wald Test
            # 语法说明: statsmodels 的 params 顺序通常是 [const_infl, const_nb, alpha]
            # 我们想检验第 1 个参数 (const_infl) 是否显著

            # R矩阵: 选择第一个参数
            # H0: beta_infl = 0
            try:
                # wald_test 会利用完整的 cov_matrix 计算
                wald_res = res.wald_test(r_matrix=np.eye(len(params))[0])
                pval_infl_multi = wald_res.pvalue
                # 注意: wald_res.pvalue 可能是标量也可能是数组，取标量
                if hasattr(pval_infl_multi, 'item'):
                    pval_infl_multi = pval_infl_multi.item()
            except Exception as e:
                # 如果 Wald Test 失败（通常因为协方差矩阵坏了），直接 Fail
                return {'status': 'Fail', 'reason': f'Multivariate Wald Error: {str(e)}'}

            # ==========================================================
            # 3. 严格判据
            # ==========================================================

            # 如果 Pi 很大 (>0.1) 但多变量 Wald 检验说不显著 (>0.05)
            # 说明虽然点估计大，但考虑到参数相关性，这可能只是噪声
            if hat_pi > 0.1 and pval_infl_multi > 0.05:
                return {
                    'status': 'Fail',
                    'reason': f'Strict Filter: Insignificant Pi (Wald p={pval_infl_multi:.3f})'
                }

            return {
                'status': 'Pass',
                'params': {'pi': hat_pi, 'alpha': hat_alpha},
                'stats': {'p_infl': pval_infl_multi, 'cond_num': cond_num},
                'reason': 'Pass'
            }

        except Exception as e:
            return {'status': 'Fail', 'reason': f'Error: {str(e)}'}
