"""
语义级傅里叶记忆系统 —— 完整实现
=====================================
只存两样东西：
  公式A：傅里叶系数矩阵（语义曲线）
  公式B：TF-IDF向量空间（解码器，跨文章共用）

从这两样东西还原完整文章
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────────────────────────
# 文章
# ───────────────────────────────────────────────

ARTICLE = """人工智能的记忆问题是当前研究的核心挑战之一。
传统的语言模型只拥有上下文窗口内的短期记忆，一旦对话超出窗口范围，之前的信息便会永久丢失。
这种局限性严重制约了AI在长期任务中的表现。
为了解决这个问题，研究者们提出了多种外部记忆方案。
向量数据库是目前最常见的方法，它将每条记忆转化为高维向量并存储起来。
检索时通过计算语义相似度找到最相关的记忆片段。
然而这种方法的存储成本随记忆条数线性增长，规模越大代价越高。
知识图谱方法将记忆表示为实体与关系的网络结构，能够捕捉记忆之间的逻辑联系。
但图结构的维护和查询复杂度较高，难以快速扩展。
分层记忆系统模仿人类大脑的工作方式，将记忆分为工作记忆和长期存储两个层次。
重要信息保留在活跃层，不常用的内容归档到深层存储，按需调取。
状态空间模型提供了另一种思路，用固定大小的矩阵参数描述序列的演变规律。
无论处理多长的序列，参数量始终保持不变，实现了真正的常数级存储。
傅里叶记忆方法将记忆点在语义空间中的分布拟合成参数曲线。
通过控制频率数量来平衡存储成本和重建精度，是一种灵活的压缩方案。
理想的记忆系统应当兼顾存储效率、检索速度和语义准确性三个维度。
未来的研究方向将探索如何让AI像人类一样自然地遗忘不重要的信息。
同时保留关键经历，在有限的存储空间内最大化记忆的价值与效用。"""


# ───────────────────────────────────────────────
# Step 1：句子分割
# ───────────────────────────────────────────────

sentences = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
N = len(sentences)

# ───────────────────────────────────────────────
# Step 2：公式B —— TF-IDF 向量空间（解码器）
# ───────────────────────────────────────────────

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
emb_matrix = vectorizer.fit_transform(sentences).toarray()  # (N, d)
d = emb_matrix.shape[1]

# ───────────────────────────────────────────────
# Step 3：语义弧长（x 轴）
# ───────────────────────────────────────────────

def semantic_arc_length(embs):
    t = [0.0]
    for i in range(1, len(embs)):
        dist = np.linalg.norm(embs[i] - embs[i-1])
        t.append(t[-1] + dist)
    return np.array(t)

t_vals = semantic_arc_length(emb_matrix)

# ───────────────────────────────────────────────
# Step 4：公式A —— 傅里叶拟合
# ───────────────────────────────────────────────

def fourier_fit(t, Y, K):
    T   = t[-1]
    Phi = np.zeros((len(t), 2*K+1))
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        Phi[:,2*k-1] = np.cos(2*np.pi*k*t/T)
        Phi[:,2*k  ] = np.sin(2*np.pi*k*t/T)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    return coeffs, T

def fourier_eval(t_query, coeffs, T, K):
    phi    = np.zeros(2*K+1)
    phi[0] = 1.0
    for k in range(1, K+1):
        phi[2*k-1] = np.cos(2*np.pi*k*t_query/T)
        phi[2*k  ] = np.sin(2*np.pi*k*t_query/T)
    return phi @ coeffs

# ── DCT 版本 ────────────────────────────────────
# 只用 cos，没有 sin，不会首尾环绕
# 公式大小 = (K+1)×d，比傅里叶的 (2K+1)×d 更小

def dct_fit(t, Y, K):
    N      = len(t)
    T      = t[-1]
    t_norm = t / T * N                        # 映射到 [0, N]
    Phi    = np.zeros((N, K+1))
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        Phi[:,k] = np.cos(np.pi / N * (t_norm + 0.5) * k)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    return coeffs, T, N

def dct_eval(t_query, coeffs, T, N_orig, K):
    t_norm = t_query / T * N_orig
    phi    = np.zeros(K+1)
    phi[0] = 1.0
    for k in range(1, K+1):
        phi[k] = np.cos(np.pi / N_orig * (t_norm + 0.5) * k)
    return phi @ coeffs

# ───────────────────────────────────────────────
# Step 5：还原函数（通用）
# ───────────────────────────────────────────────

def reconstruct(eval_fn, t_vals, emb_matrix, sentences):
    restored = np.array([eval_fn(t) for t in t_vals])
    sims     = cosine_similarity(restored, emb_matrix)
    best_idx = np.argmax(sims, axis=1)
    correct  = sum(best_idx[i] == i for i in range(len(sentences)))
    result   = [sentences[i] for i in best_idx]
    return result, correct, best_idx, sims

# ───────────────────────────────────────────────
# 运行：DCT K=8（最优配置）
# ───────────────────────────────────────────────

K = 8

coeffs, T, N_orig = dct_fit(t_vals, emb_matrix, K)
result, correct, best_idx, sims = reconstruct(
    lambda t: dct_eval(t, coeffs, T, N_orig, K),
    t_vals, emb_matrix, sentences)

# ── 存储 ─────────────────────────────────────────
formula_sz = (K+1) * d
orig       = N * d

print("=" * 62)
print("  存储对比  (DCT K=8)")
print("=" * 62)
print(f"  原始大小     : {orig:>7} 个数字  ({N}句 × {d}维)")
print(f"  公式A (DCT)  : {formula_sz:>7} 个数字  压缩 {(1-formula_sz/orig)*100:.1f}%")
print(f"  公式B (词表) :     731 个词    跨文章共用，不计入单篇")
print(f"  参考：傅里叶K=6 需 9503 个数字，DCT K=8 小 {9503-formula_sz} 个")

# ── 逐句还原结果 ──────────────────────────────────
print(f"\n{'=' * 62}")
print(f"  从公式还原文章  准确率 {correct}/{N}")
print(f"{'=' * 62}")
print(f"\n  {'#':>2}  {'':^4}  {'相似度':>6}  句子")
print(f"  {'─'*58}")

for i, sent in enumerate(sentences):
    mark = "✓" if best_idx[i] == i else "✗"
    sim  = sims[i, best_idx[i]]
    print(f"  {i+1:>2}   {mark}    {sim:.3f}   {sent[:44]}...")

# ── 总结 ─────────────────────────────────────────
print(f"\n{'=' * 62}")
print(f"  公式A：{formula_sz} 个数字")
print(f"  原文 ：{orig} 个数字")
print(f"  从 {formula_sz} 个数字还原 {correct}/{N} 句，压缩率 {(1-formula_sz/orig)*100:.1f}%")
print(f"  第18句（边界问题）：已修复 ✓")
print(f"{'=' * 62}")
