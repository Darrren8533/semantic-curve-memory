"""
Fourier Memory System
=====================
用傅里叶级数做 Line 2 的公式
核心优势：存储大小固定，不随记忆条数增长

Line 1: 稀疏点 {(t_i, 词_i)}
Line 2: f(t) = a0 + sum_k [ ak*cos(2πkt/T) + bk*sin(2πkt/T) ]
        只需存 (2K+1) × d 个系数，K 是你选的频率数，固定的

存储:  系数矩阵 (固定大小) + t 值 (稀疏) + 词标签
"""

import numpy as np
from scipy.spatial.distance import cosine

WORD_EMBEDDINGS = {
    "我":   np.array([0.1, 0.1, 0.1, 0.1,  0.2, 0.1, 0.1,  0.9, 0.8, 0.7]),
    "你":   np.array([0.1, 0.1, 0.1, 0.1,  0.1, 0.2, 0.1,  0.8, 0.9, 0.6]),
    "他":   np.array([0.1, 0.1, 0.1, 0.2,  0.2, 0.1, 0.2,  0.7, 0.7, 0.9]),
    "想":   np.array([0.2, 0.1, 0.1, 0.1,  0.8, 0.7, 0.3,  0.3, 0.2, 0.1]),
    "吃":   np.array([0.3, 0.2, 0.1, 0.2,  0.9, 0.8, 0.2,  0.2, 0.1, 0.1]),
    "买":   np.array([0.2, 0.1, 0.2, 0.1,  0.7, 0.9, 0.3,  0.2, 0.2, 0.1]),
    "跑":   np.array([0.1, 0.1, 0.1, 0.2,  0.9, 0.6, 0.9,  0.2, 0.1, 0.1]),
    "苹果": np.array([0.9, 0.8, 0.7, 0.6,  0.1, 0.1, 0.2,  0.1, 0.1, 0.1]),
    "香蕉": np.array([0.8, 0.9, 0.6, 0.7,  0.1, 0.1, 0.1,  0.1, 0.1, 0.2]),
    "咖啡": np.array([0.6, 0.5, 0.9, 0.8,  0.1, 0.2, 0.1,  0.1, 0.2, 0.1]),
    "水":   np.array([0.5, 0.4, 0.8, 0.7,  0.1, 0.1, 0.2,  0.1, 0.1, 0.1]),
    "茶":   np.array([0.5, 0.5, 0.8, 0.8,  0.1, 0.1, 0.1,  0.1, 0.1, 0.2]),
}


# ─────────────────────────────────────────────────────────────
#  傅里叶公式：拟合 & 求值
# ─────────────────────────────────────────────────────────────

def _design_matrix(t_vals, T, n_freqs):
    """构建傅里叶设计矩阵 Φ，形状 (n, 2K+1)"""
    n = len(t_vals)
    Phi = np.zeros((n, 2 * n_freqs + 1))
    Phi[:, 0] = 1.0
    for k in range(1, n_freqs + 1):
        Phi[:, 2*k-1] = np.cos(2 * np.pi * k * t_vals / T)
        Phi[:, 2*k  ] = np.sin(2 * np.pi * k * t_vals / T)
    return Phi

def fit_fourier(t_vals, embeddings, n_freqs):
    """
    最小二乘法拟合傅里叶系数
    返回 coeffs: 形状 (2K+1, d)  ← 这就是"公式"
    """
    T = t_vals[-1]
    Phi = _design_matrix(t_vals, T, n_freqs)
    # 最小二乘：Phi @ coeffs ≈ embeddings
    coeffs, _, _, _ = np.linalg.lstsq(Phi, embeddings, rcond=None)
    return coeffs, T

def eval_fourier(t, coeffs, T, n_freqs):
    """在 t 处求公式的值"""
    phi = np.zeros(2 * n_freqs + 1)
    phi[0] = 1.0
    for k in range(1, n_freqs + 1):
        phi[2*k-1] = np.cos(2 * np.pi * k * t / T)
        phi[2*k  ] = np.sin(2 * np.pi * k * t / T)
    return phi @ coeffs


# ─────────────────────────────────────────────────────────────
#  记忆系统
# ─────────────────────────────────────────────────────────────

class FourierMemory:

    def __init__(self, n_freqs: int = 3):
        """
        n_freqs: 傅里叶频率数 K
        存储大小 = (2K+1) × d，与记忆条数无关
        K 越大精度越高，但存储也越大（你自己定预算）
        """
        self.n_freqs   = n_freqs
        self.words     = []
        self.t_values  = None   # ← 存储: 每个词的弧长坐标
        self.coeffs    = None   # ← 存储: 傅里叶系数矩阵（固定大小）
        self.T         = None   # ← 存储: 总弧长（1个数）
        self._raw_embs = None   # 临时用，不算存储

    def memorize(self, sentence: list[str]):
        self.words    = sentence
        self._raw_embs = np.array([WORD_EMBEDDINGS[w] for w in sentence])
        self.t_values = self._arc_length(self._raw_embs)
        self.coeffs, self.T = fit_fourier(self.t_values, self._raw_embs, self.n_freqs)
        self._print_memorize()

    def _arc_length(self, embs):
        t = [0.0]
        for i in range(1, len(embs)):
            t.append(t[-1] + np.linalg.norm(embs[i] - embs[i-1]))
        return np.array(t)

    def _print_memorize(self):
        print(f"\n  memorized: {' -> '.join(self.words)}")
        print(f"  t values : {np.round(self.t_values, 3)}")
        d = self._raw_embs.shape[1]
        formula_size = (2 * self.n_freqs + 1) * d
        print(f"  formula  : ({2*self.n_freqs+1} freqs) x ({d} dims) = {formula_size} numbers  [FIXED]")

    def add_one(self, new_word: str):
        """新记忆自适应加入，公式大小不变"""
        new_emb = WORD_EMBEDDINGS[new_word]
        new_t   = self.t_values[-1] + np.linalg.norm(new_emb - self._raw_embs[-1])

        self.words.append(new_word)
        self._raw_embs = np.vstack([self._raw_embs, new_emb])
        self.t_values  = np.append(self.t_values, new_t)
        # 用新数据重新拟合 —— 系数大小不变，只是值更新
        self.coeffs, self.T = fit_fourier(self.t_values, self._raw_embs, self.n_freqs)
        print(f"  + added '{new_word}' at t={new_t:.3f}  |  formula size unchanged")

    def retrieve(self, query_word: str, top_k: int = 3):
        query_emb = WORD_EMBEDDINGS[query_word]
        scores = []
        for i, word in enumerate(self.words):
            # 从公式还原 embedding（Line1 和 Line2 的交叉点）
            restored = eval_fourier(self.t_values[i], self.coeffs, self.T, self.n_freqs)
            sim = 1 - cosine(query_emb, restored)
            scores.append((sim, word))
        scores.sort(reverse=True)

        print(f"\n  query: '{query_word}'")
        for rank, (sim, word) in enumerate(scores[:top_k], 1):
            bar = "█" * int(sim * 20)
            print(f"    {rank}. '{word}'  {sim:.3f}  {bar}")

    def storage_report(self, compare_counts: list[int] = None):
        d          = self._raw_embs.shape[1]
        formula_sz = (2 * self.n_freqs + 1) * d   # 固定
        t_sz       = len(self.t_values)

        print(f"\n  {'─'*50}")
        print(f"  Storage (K={self.n_freqs} freqs, d={d} dims)")
        print(f"  {'─'*50}")
        print(f"  Formula (coeffs)  : {formula_sz:>5}  <- FIXED no matter how many words")
        print(f"  t values          : {t_sz:>5}  (one per word, tiny)")
        print(f"  {'─'*50}")

        # 不同词数下的对比
        counts = compare_counts or [5, 10, 50, 100, 500]
        print(f"  {'words':>6}  {'original':>10}  {'fourier':>10}  {'saving':>8}")
        print(f"  {'─'*40}")
        for n in counts:
            orig = n * d
            four = formula_sz + n   # formula固定 + n个t值
            saving = (1 - four/orig) * 100
            tag = " <- break-even" if abs(saving) < 5 else ""
            print(f"  {n:>6}  {orig:>10}  {four:>10}  {saving:>7.1f}%{tag}")
        print(f"  {'─'*50}")
        print(f"  After ~{formula_sz} words, Fourier memory is always smaller")


# ─────────────────────────────────────────────────────────────
#  演示
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 52)
    print("  Fourier Memory  (K=3 frequencies)")
    print("=" * 52)

    mem = FourierMemory(n_freqs=3)

    print("\n[1] memorize")
    mem.memorize(["我", "想", "吃", "苹果", "买", "咖啡"])

    print("\n[2] retrieve")
    mem.retrieve("香蕉")   # -> 苹果
    mem.retrieve("水")     # -> 咖啡
    mem.retrieve("你")     # -> 我

    print("\n[3] add new memory (formula size unchanged)")
    mem.add_one("茶")
    mem.add_one("香蕉")
    mem.add_one("他")
    mem.retrieve("你")     # -> 我 / 他

    print("\n[4] storage report")
    mem.storage_report()
