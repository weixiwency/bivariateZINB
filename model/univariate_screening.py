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

        # 1. 硬过滤
        if np.sum(y > 0) < 5:
            return {'status': 'Fail', 'reason': 'Too few cells (<5)'}
        if np.mean(y) < 0.01:
            return {'status': 'Fail', 'reason': 'Mean too low'}

        # 2. 准备数据
        # 因为我们只拟合分布，没有协变量，所以 exog 是全 1 的截距项
        n_samples = len(y)
        exog = np.ones((n_samples, 1))  # NB 部分的截距
        exog_infl = np.ones((n_samples, 1))  # Zero-inflation 部分的截距

        # 3. 建立模型 (ZINB-P, p=2 对应 NB2)
        # inflation='logit' 是默认且标准的
        try:
            model = ZeroInflatedNegativeBinomialP(
                endog=y,
                exog=exog,
                exog_infl=exog_infl,
                inflation='logit',
                p=2
            )

            # 4. 拟合
            # method='bfgs' 通常比较稳健，disp=0 不打印日志
            res = model.fit(method='bfgs', maxiter=100, disp=0)

            # 检查收敛状态
            if not res.mle_retvals['converged']:
                return {'status': 'Fail', 'reason': 'Optimizer not converged'}

        except Exception as e:
            return {'status': 'Fail', 'reason': f'Statsmodels Error: {str(e)}'}

        # 5. 提取参数和统计量 (Wald Test 核心)
        try:
            # statsmodels 的参数顺序通常是:
            # [params_infl(截距), params_count(截距), alpha]
            # 我们可以通过 param_names 来确认，但在纯截距模型中：
            # res.params[0] -> inflation intercept (logit pi)
            # res.params[1] -> count intercept (log mu)
            # res.params[2] -> alpha (dispersion)

            params = res.params
            bse = res.bse  # 标准误 (Standard Error)

            # --- 转换参数到物理意义 ---
            beta_infl = params[0]
            beta_count = params[1]
            alpha_val = params[2]

            # logit(pi) = beta -> pi = 1 / (1 + exp(-beta))
            hat_pi = 1.0 / (1.0 + np.exp(-beta_infl))

            # log(mu) = beta -> mu = exp(beta)
            hat_mu = np.exp(beta_count)

            # statsmodels 的 alpha 与我们之前的 m 关系: alpha = 1/m
            # 但为了防止除零，我们直接记录 alpha
            hat_alpha = alpha_val

            # --- 提取标准误 (SE) ---
            se_beta_infl = bse[0]
            se_beta_count = bse[1]
            se_alpha = bse[2]

            # --- 质检判据 (Wald Test Logic) ---

            # 判据 A: 检查 SE 是否为 NaN 或 无穷大 (Hessian 奇异)
            if np.any(np.isnan(bse)) or np.any(np.isinf(bse)):
                return {'status': 'Fail', 'reason': 'Singular Hessian (NaN SE)'}

            # 判据 B: 参数不可识别 (Unidentifiable)
            # 如果截距项的标准误极大 (比如 > 100)，说明数据撑不起这个模型
            # 尤其是 Alpha (离散度)
            if se_alpha > 50:
                return {'status': 'Fail', 'reason': 'High SE on Alpha'}

            # 判据 C: 边界检查
            # 如果 pi 极其接近 1，说明全是 0，没意义
            if hat_pi > 0.995:
                return {'status': 'Fail', 'reason': 'Pi ~ 1 (All Zero)'}

            # 判据 D: Theta (Alpha) 边界
            # 如果 alpha 极小 (接近 Poisson) 或 极大 (Over-dispersed)，视情况而定
            # 这里主要防 alpha < 0 (虽然 statsmodels 通常限制 >0)
            if hat_alpha < 1e-6:
                return {'status': 'Fail', 'reason': 'Alpha -> 0 (Poisson)'}

            # 判据 E: Pi 的 SE 检查
            # 如果 Pi 估计值 > 0.1 (认为有零膨胀)，但 SE 巨大，说明分不清是 NB 的尾巴还是 Pi
            # 这里我们检查 Logit 尺度上的 SE
            if hat_pi > 0.1 and se_beta_infl > 10:
                return {'status': 'Fail', 'reason': 'High SE on Inflation'}

            return {
                'status': 'Pass',
                'params': {'pi': hat_pi, 'mu': hat_mu, 'alpha': hat_alpha},
                'raw_params': {'logit_pi': beta_infl, 'log_mu': beta_count},
                'se': {'se_logit_pi': se_beta_infl, 'se_log_mu': se_beta_count, 'se_alpha': se_alpha}
            }

        except Exception as e:
            return {'status': 'Fail', 'reason': f'Result Extraction Error: {str(e)}'}