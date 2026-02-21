"""
真实 embedding 验证
====================
用 sentence-transformers 替换 TF-IDF
看 DCT 记忆系统在 768 维真实 embedding 下的表现

对比：
  TF-IDF (weak)  vs  sentence-transformers (strong)
  两种 embedding 下，DCT + 多起点检索的命中率和压缩率
"""

import torch  # 必须先 import，让 DLL 在正确时机初始化
import numpy as np
import time
from scipy.optimize import minimize_scalar
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ───────────────────────────────────────────────
# 数据
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

QUERIES = [
    ("存储成本随条数增长",       [6]),
    ("固定大小参数不随序列增长",  [11, 12]),
    ("图结构实体关系",            [7, 8]),
    ("工作记忆长期存储分层",      [9, 10]),
    ("频率控制压缩精度",          [13, 14]),
    ("遗忘不重要信息",            [16, 17]),
]

sentences = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
N = len(sentences)

# ───────────────────────────────────────────────
# 加载模型（多语言，支持中文）
# ───────────────────────────────────────────────

print("加载 sentence-transformers 模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("模型加载完成\n")

# 编码句子
E = model.encode(sentences, show_progress_bar=False)   # (18, 384)
d = E.shape[1]
print(f"embedding 维度: {d}")

# ───────────────────────────────────────────────
# DCT 工具（通用）
# ───────────────────────────────────────────────

def arc_length(E):
    t = [0.0]
    for i in range(1, len(E)):
        t.append(t[-1] + np.linalg.norm(E[i] - E[i-1]))
    return np.array(t)

def dct_fit(t, Y, K):
    n = len(t); T = t[-1]; tn = t / T * n
    P = np.zeros((n, K+1)); P[:,0] = 1.0
    for k in range(1, K+1):
        P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    c, _, _, _ = np.linalg.lstsq(P, Y, rcond=None)
    return c, T, n

def dct_eval(tq, c, T, n, K):
    tn = tq / T * n; phi = np.zeros(K+1); phi[0] = 1.0
    for k in range(1, K+1): phi[k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return phi @ c

def dct_eval_batch(t_arr, c, T, n, K):
    tn = t_arr / T * n
    P  = np.zeros((len(t_arr), K+1)); P[:,0] = 1.0
    for k in range(1, K+1): P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return P @ c

# ───────────────────────────────────────────────
# 多起点检索
# ───────────────────────────────────────────────

def multistart_retrieval(q_emb, coeffs, T_c, N_c, t_vals, restored, K,
                          n_starts=6, window_ratio=0.15, top_k=2):
    window = T_c * window_ratio

    def dist(t):
        return np.sum((dct_eval(t, coeffs, T_c, N_c, K) - q_emb) ** 2)

    starts = np.linspace(0, T_c, n_starts + 2)[1:-1]
    all_t_star = []
    for s in starts:
        w = T_c / n_starts
        res = minimize_scalar(dist, bounds=(max(0, s-w), min(T_c, s+w)),
                              method='bounded')
        all_t_star.append(res.x)

    cand_set = set()
    for ts in all_t_star:
        for i, t in enumerate(t_vals):
            if abs(t - ts) < window:
                cand_set.add(i)

    candidates = sorted(cand_set) or [np.argmin([min(abs(t-ts) for ts in all_t_star)
                                                  for t in t_vals])]
    sims = cosine_similarity(q_emb.reshape(1,-1), restored[candidates])[0]
    top  = np.argsort(sims)[::-1][:top_k]
    return [candidates[i] for i in top], sims[top].tolist(), len(candidates)

def linear_scan(q_emb, restored, top_k=2):
    sims = cosine_similarity(q_emb.reshape(1,-1), restored)[0]
    idx  = np.argsort(sims)[::-1][:top_k]
    return idx.tolist(), sims[idx].tolist()

# ───────────────────────────────────────────────
# 在不同 K 下测试
# ───────────────────────────────────────────────

t_vals = arc_length(E)
Q      = len(QUERIES)
q_embs = model.encode([q for q, _ in QUERIES], show_progress_bar=False)

print(f"\n{'='*65}")
print(f"  sentence-transformers ({d}维) + DCT + 多起点检索")
print(f"{'='*65}")
print(f"\n  原始大小 = {N} × {d} = {N*d} 个数字\n")
print(f"  {'K':>4}  {'公式大小':>8}  {'压缩率':>8}  {'命中率':>8}  {'平均检查点':>10}  {'减少比较':>8}")
print(f"  {'─'*58}")

best_result = None

for K in [4, 6, 8, 10, 12]:
    coeffs, T_c, N_c = dct_fit(t_vals, E, K)
    restored         = dct_eval_batch(t_vals, coeffs, T_c, N_c, K)

    hits = 0; checked = []
    for (q_text, expected), q_emb in zip(QUERIES, q_embs):
        idx, _, n_cand = multistart_retrieval(q_emb, coeffs, T_c, N_c,
                                               t_vals, restored, K)
        if any(i in expected for i in idx): hits += 1
        checked.append(n_cand)

    formula_sz = (K+1) * d
    compress   = (1 - formula_sz / (N*d)) * 100
    hit_rate   = hits / Q * 100
    avg_check  = np.mean(checked)
    reduce     = (1 - avg_check / N) * 100

    tag = ""
    if hits == Q and (best_result is None or formula_sz < best_result[0]):
        best_result = (formula_sz, K, compress, avg_check, reduce)
        tag = " ← 最优"

    print(f"  K={K:>2}  {formula_sz:>8}  {compress:>7.1f}%  {hit_rate:>7.1f}%  {avg_check:>10.1f}  {reduce:>7.1f}%{tag}")

# ───────────────────────────────────────────────
# 最优 K 的详细结果
# ───────────────────────────────────────────────

best_K = best_result[1]
coeffs, T_c, N_c = dct_fit(t_vals, E, best_K)
restored         = dct_eval_batch(t_vals, coeffs, T_c, N_c, best_K)

print(f"\n{'='*65}")
print(f"  最优 K={best_K} 详细检索结果")
print(f"{'='*65}")
print(f"\n  {'查询':24}  {'多起点':^8}  {'线性扫描':^8}  检查点数  结果句子")
print(f"  {'─'*62}")

for (q_text, expected), q_emb in zip(QUERIES, q_embs):
    m_idx, m_sims, n_cand = multistart_retrieval(q_emb, coeffs, T_c, N_c,
                                                   t_vals, restored, best_K)
    l_idx, l_sims         = linear_scan(q_emb, restored)

    mh = "✓" if any(i in expected for i in m_idx) else "✗"
    lh = "✓" if any(i in expected for i in l_idx) else "✗"
    print(f"  {q_text:24}  {mh:^8}  {lh:^8}  {n_cand:>4}/{N}    句{[i+1 for i in m_idx]}")

# ───────────────────────────────────────────────
# TF-IDF vs sentence-transformers 最终对比
# ───────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  TF-IDF vs sentence-transformers 最终对比")
print(f"{'='*65}")
print(f"""
  {'':30}  {'TF-IDF':>12}  {'sentence-T':>12}
  {'─'*58}
  {'embedding 维度':30}  {'731':>12}  {d:>12}
  {'公式大小 (最优K)':30}  {'6579':>12}  {best_result[0]:>12}
  {'原始大小':30}  {'13158':>12}  {N*d:>12}
  {'压缩率':30}  {'50.0%':>12}  {best_result[2]:>11.1f}%
  {'检索命中率':30}  {'100%':>12}  {'100%':>12}
  {'平均检查点数':30}  {'10.7':>12}  {best_result[3]:>12.1f}
  {'─'*58}
  sentence-transformers embedding 维度更低({d}维 < 731维)
  压缩率相近，命中率相同，但语义理解能力强很多
""")
