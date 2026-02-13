import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
import warnings

# 忽略数值计算告警
warnings.filterwarnings('ignore')

class Smart_Initializer:
    @staticmethod
    def init_pis_by_counts(y1, y2):
        """初始化 Pi"""
        N = len(y1)
        n1 = np.sum((y1 > 0) & (y2 > 0))  # Both Active
        n2 = np.sum((y1 > 0) & (y2 == 0))  # G1 Only
        n3 = np.sum((y1 == 0) & (y2 > 0))  # G2 Only
        n4 = np.sum((y1 == 0) & (y2 == 0))  # Both Zero

        counts = np.array([n1, n2, n3, n4]) + 1.0  # 加平滑
        return counts / np.sum(counts)

    @staticmethod
    def get_marginal_zinb_params(y):
        """
        单基因 ZINB 矩估计用于 m 和 theta 初始化
        """
        n = len(y)
        mean_val = np.mean(y)
        var_val = np.var(y)
        p0 = np.sum(y == 0) / n

        # 估计 pi
        pi_hat = min(p0, 0.9)

        # 还原 NB 均值和方差
        mu_nb = mean_val / (1 - pi_hat + 1e-6)
        inflation_term = pi_hat * (1 - pi_hat) * (mu_nb ** 2)
        var_nb = (var_val - inflation_term) / (1 - pi_hat + 1e-6)

        # 计算 m 和 theta
        if var_nb > mu_nb:
            m_init = (var_nb - mu_nb) / (mu_nb ** 2 + 1e-6)
        else:
            m_init = 1.0

        theta_init = (m_init * mu_nb) / (1 + m_init * mu_nb)
        return m_init, theta_init


class BZINB_Model:
    def __init__(self, tol=1e-4, max_iter=50):
        self.tol = tol
        self.max_iter = max_iter
        self.params = {}
        self.pis = None

    def nb_logpmf(self, y, m, theta):
        """单变量 NB Log"""
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        m = np.clip(m, 1e-6, 1e4)
        r = 1.0 / m
        coeff = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        return coeff + y * np.log(theta) + r * np.log(1 - theta)

    def get_famoye_term(self, y, m, t):
        """修正项辅助变量"""
        e_inv = np.exp(-1)
        base = (1 - t) / (1 - t * e_inv)
        c = base ** (1 / m)
        return np.exp(-y) - c

    def bnb_logpmf(self, y1, y2, m1, t1, m2, t2, lam):
        """双变量 BNB Log"""
        log_p1 = self.nb_logpmf(y1, m1, t1)
        log_p2 = self.nb_logpmf(y2, m2, t2)
        term1 = self.get_famoye_term(y1, m1, t1)
        term2 = self.get_famoye_term(y2, m2, t2)
        correction = np.maximum(1 + lam * term1 * term2, 1e-10)
        return log_p1 + log_p2 + np.log(correction)

    def fit(self, y1, y2):
        """
        输入: 两个一维数组 y1, y2 (长度为 N)
        输出: 拟合好的参数字典
        """
        # 1.初始化
        self.pis = Smart_Initializer.init_pis_by_counts(y1, y2)
        m1_in, t1_in = Smart_Initializer.get_marginal_zinb_params(y1)
        m2_in, t2_in = Smart_Initializer.get_marginal_zinb_params(y2)

        self.params = {
            'm1': m1_in, 't1': t1_in,
            'm2': m2_in, 't2': t2_in,
            'lam': 0.0
        }

        # 2.EM迭代
        log_lik_history = []

        for iteration in range(self.max_iter):
            # E-step
            lp1 = self.bnb_logpmf(y1, y2, self.params['m1'], self.params['t1'],
                                  self.params['m2'], self.params['t2'], self.params['lam'])
            p1 = np.exp(lp1) * self.pis[0]

            p2 = np.exp(self.nb_logpmf(y1, self.params['m1'], self.params['t1'])) * (y2 == 0) * self.pis[1]
            p3 = np.exp(self.nb_logpmf(y2, self.params['m2'], self.params['t2'])) * (y1 == 0) * self.pis[2]
            p4 = ((y1 == 0) & (y2 == 0)).astype(float) * self.pis[3]

            total_prob = p1 + p2 + p3 + p4 + 1e-20
            gamma = np.vstack([p1, p2, p3, p4]) / total_prob
            gamma = gamma.T

            # M-step
            self.pis = gamma.mean(axis=0)

            # 准备参数
            m1, t1 = self.params['m1'], self.params['t1']
            m2, t2 = self.params['m2'], self.params['t2']
            lam = self.params['lam']

            # 优化 Gene 1
            w1 = gamma[:, 0] + gamma[:, 1]
            w_corr = gamma[:, 0]
            term2_fixed = self.get_famoye_term(y2, m2, t2)

            def loss_g1(p):
                ll_nb = np.sum(w1 * self.nb_logpmf(y1, p[0], p[1]))
                term1 = self.get_famoye_term(y1, p[0], p[1])
                corr = np.maximum(1 + lam * term1 * term2_fixed, 1e-10)
                return -(ll_nb + np.sum(w_corr * np.log(corr)))

            res1 = minimize(loss_g1, [m1, t1], bounds=[(1e-4, 20), (1e-4, 0.999)], method='L-BFGS-B')
            self.params['m1'], self.params['t1'] = res1.x

            # 优化 Gene 2
            w2 = gamma[:, 0] + gamma[:, 2]
            term1_fixed = self.get_famoye_term(y1, self.params['m1'], self.params['t1'])

            def loss_g2(p):
                ll_nb = np.sum(w2 * self.nb_logpmf(y2, p[0], p[1]))
                term2 = self.get_famoye_term(y2, p[0], p[1])
                corr = np.maximum(1 + lam * term1_fixed * term2, 1e-10)
                return -(ll_nb + np.sum(w_corr * np.log(corr)))

            res2 = minimize(loss_g2, [m2, t2], bounds=[(1e-4, 20), (1e-4, 0.999)], method='L-BFGS-B')
            self.params['m2'], self.params['t2'] = res2.x

            # 优化 Lambda
            term1_final = self.get_famoye_term(y1, self.params['m1'], self.params['t1'])
            term2_final = self.get_famoye_term(y2, self.params['m2'], self.params['t2'])

            def loss_lam(l):
                corr = np.maximum(1 + l[0] * term1_final * term2_final, 1e-10)
                return -np.sum(gamma[:, 0] * np.log(corr))

            res_lam = minimize(loss_lam, [lam], bounds=[(-10, 10)], method='L-BFGS-B')
            self.params['lam'] = res_lam.x[0]

            # 收敛检查
            curr_ll = np.sum(np.log(total_prob))
            log_lik_history.append(curr_ll)
            if iteration > 0 and abs(log_lik_history[-1] - log_lik_history[-2]) < self.tol:
                break

        return self.params, self.pis


# Test
if __name__ == "__main__":
    # 随便的一个示例（输入两个基因的向量，输出参数的dict）
    y1 = np.array([0, 0, 5, 10, 0, 20, 0, 3, 4, 0])
    y2 = np.array([0, 0, 4, 12, 0, 0, 5, 2, 0, 1])

    model = BZINB_Model()

    params, pis = model.fit(y1, y2)
