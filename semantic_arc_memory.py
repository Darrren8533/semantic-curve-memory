"""
Semantic Arc-Length Memory System
==================================
Line 1: 稀疏的记忆点 (词 + embedding坐标)
Line 2: 穿过所有点的参数曲线，x轴 = 语义弧长
记忆   = Line1 和 Line2 的交集 (= Line1 的所有点)
存储   = 曲线公式参数 + t值 + 词标签  (不存原始embedding)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cosine


# ─────────────────────────────────────────────
# 手工定义有语义意义的 embedding（10维，模拟真实情况）
# 食物类词：维度0~3 权重高
# 动作类词：维度4~6 权重高
# 人称类词：维度7~9 权重高
# ─────────────────────────────────────────────
WORD_EMBEDDINGS = {
    "我":   np.array([0.1, 0.1, 0.1, 0.1,  0.2, 0.1, 0.1,  0.9, 0.8, 0.7]),
    "你":   np.array([0.1, 0.1, 0.1, 0.1,  0.1, 0.2, 0.1,  0.8, 0.9, 0.6]),
    "想":   np.array([0.2, 0.1, 0.1, 0.1,  0.8, 0.7, 0.3,  0.3, 0.2, 0.1]),
    "吃":   np.array([0.3, 0.2, 0.1, 0.2,  0.9, 0.8, 0.2,  0.2, 0.1, 0.1]),
    "买":   np.array([0.2, 0.1, 0.2, 0.1,  0.7, 0.9, 0.3,  0.2, 0.2, 0.1]),
    "苹果": np.array([0.9, 0.8, 0.7, 0.6,  0.1, 0.1, 0.2,  0.1, 0.1, 0.1]),
    "香蕉": np.array([0.8, 0.9, 0.6, 0.7,  0.1, 0.1, 0.1,  0.1, 0.1, 0.2]),
    "咖啡": np.array([0.6, 0.5, 0.9, 0.8,  0.1, 0.2, 0.1,  0.1, 0.2, 0.1]),
    "水":   np.array([0.5, 0.4, 0.8, 0.7,  0.1, 0.1, 0.2,  0.1, 0.1, 0.1]),
    "跑":   np.array([0.1, 0.1, 0.1, 0.2,  0.9, 0.6, 0.9,  0.2, 0.1, 0.1]),
}


class SemanticArcMemory:
    """
    两条线的记忆系统：
      Line 1: 实际记忆点 {(t_i, e_i)}
      Line 2: 参数曲线 f(t) → R^d，t = 语义弧长
      记忆   = 交集（Line1 上的点）
    """

    def __init__(self):
        self.words      = []
        self._embeddings = None   # 原始 embedding，只用来建曲线，之后可以扔掉
        self.t_values   = None    # ← 存储: x轴坐标（弧长）
        self.curve      = None    # ← 存储: 曲线公式（CubicSpline 参数）
        self._dim       = 0

    # ── 存记忆 ──────────────────────────────────
    def memorize(self, sentence: list[str]):
        """输入词列表，建立两条线并存储"""
        self.words = sentence
        self._embeddings = np.array([WORD_EMBEDDINGS[w] for w in sentence])
        self._dim = self._embeddings.shape[1]

        # Line 1 的 x 坐标 = 语义弧长
        self.t_values = self._compute_arc_length(self._embeddings)

        # Line 2 = 拟合穿过所有点的参数曲线
        self.curve = CubicSpline(self.t_values, self._embeddings)

        print(f"\n✓ 记忆了 {len(self.words)} 个词")
        print(f"  词序列: {' → '.join(self.words)}")
        print(f"  t 值  : {np.round(self.t_values, 3)}")

    def _compute_arc_length(self, embeddings):
        """累积语义距离 → 每个词的 t 值"""
        t = [0.0]
        for i in range(1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[i - 1])
            t.append(t[-1] + dist)
        return np.array(t)

    # ── 查记忆 ──────────────────────────────────
    def retrieve(self, query_word: str, top_k: int = 3):
        """用一个词查询语义最近的记忆"""
        if query_word not in WORD_EMBEDDINGS:
            print(f"'{query_word}' 不在词表里")
            return

        query_emb = WORD_EMBEDDINGS[query_word]

        # 只在 Line 1 的交叉点（实际记忆点）上计算相似度
        scores = []
        for i, word in enumerate(self.words):
            # 从曲线公式还原 embedding（不直接用原始数据）
            restored_emb = self.curve(self.t_values[i])
            sim = 1 - cosine(query_emb, restored_emb)
            scores.append((sim, word, self.t_values[i]))

        scores.sort(reverse=True)

        print(f"\n查询词: '{query_word}'  → 最相关的记忆:")
        for rank, (sim, word, t) in enumerate(scores[:top_k], 1):
            bar = "█" * int(sim * 20)
            print(f"  {rank}. '{word}'  (t={t:.3f})  相似度={sim:.3f}  {bar}")

    # ── 新记忆自适应更新 ─────────────────────────
    def add_one(self, new_word: str):
        """新记忆进来，自适应更新曲线（不重建，只延伸）"""
        new_emb = WORD_EMBEDDINGS[new_word]
        last_emb = self._embeddings[-1]
        new_t = self.t_values[-1] + np.linalg.norm(new_emb - last_emb)

        self.words.append(new_word)
        self._embeddings = np.vstack([self._embeddings, new_emb])
        self.t_values = np.append(self.t_values, new_t)

        # 曲线用新点集重新拟合（实际可用 Kalman 增量更新，这里简化）
        self.curve = CubicSpline(self.t_values, self._embeddings)
        print(f"\n+ 新增记忆: '{new_word}'  t={new_t:.3f}  曲线已更新")

    # ── 存储效率对比 ──────────────────────────────
    def storage_report(self):
        n_words = len(self.words)
        dim     = self._dim

        original   = n_words * dim          # 存所有 embedding
        # 曲线参数：CubicSpline 每个维度存 (n-1)*4 个系数
        curve_params = (n_words - 1) * 4 * dim
        t_params     = n_words               # t 值
        labels       = n_words               # 词标签（忽略字符串大小）

        stored = curve_params + t_params

        print(f"\n{'─'*45}")
        print(f"存储对比 ({n_words} 个词, {dim} 维 embedding)")
        print(f"{'─'*45}")
        print(f"  原始存所有 embedding : {original:>6} 个数字")
        print(f"  曲线参数             : {curve_params:>6} 个数字")
        print(f"  t 值                 : {t_params:>6} 个数字")
        print(f"  合计存储             : {stored:>6} 个数字")
        print(f"  压缩比               : {original/stored:.2f}x")
        print(f"{'─'*45}")
        print(f"  注: 词数增多时曲线参数线性增长，")
        print(f"      但公式本身只需 A,B,C 矩阵（SSM思路）可做到常数大小")


# ─────────────────────────────────────────────
# 运行演示
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mem = SemanticArcMemory()

    print("=" * 45)
    print("  Step 1: 存入一句话的记忆")
    print("=" * 45)
    mem.memorize(["我", "想", "吃", "苹果"])

    print("\n" + "=" * 45)
    print("  Step 2: 查询语义相关的记忆")
    print("=" * 45)
    mem.retrieve("香蕉")    # 食物类 → 应该找到"苹果"
    mem.retrieve("吃")      # 动作类 → 应该找到"吃""想"
    mem.retrieve("你")      # 人称类 → 应该找到"我"

    print("\n" + "=" * 45)
    print("  Step 3: 加入新记忆（自适应更新）")
    print("=" * 45)
    mem.add_one("咖啡")
    mem.retrieve("水")      # 应该找到"咖啡"（都是饮品）

    print("\n" + "=" * 45)
    print("  Step 4: 存储效率")
    print("=" * 45)
    mem.storage_report()
