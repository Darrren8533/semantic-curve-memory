"""
曲线检索 vs 线性扫描
========================
你的想法：query 生成新线 → 找跟记忆曲线靠近的位置 → 只看附近的交叉点

Step 1：找 t* = 曲线上离 query 最近的点
Step 2：只看 t* 附近的记忆交叉点
不用翻全部
"""

import numpy as np
import time
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vec = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
E   = vec.fit_transform(sentences).toarray()
d   = E.shape[1]

# ───────────────────────────────────────────────
# DCT 工具
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

def dct_eval(t_query, c, T, n, K):
    tn  = t_query / T * n
    phi = np.zeros(K+1); phi[0] = 1.0
    for k in range(1, K+1):
        phi[k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return phi @ c

def dct_eval_batch(t_arr, c, T, n, K):
    tn = t_arr / T * n
    P  = np.zeros((len(t_arr), K+1)); P[:,0] = 1.0
    for k in range(1, K+1):
        P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return P @ c

# 建立 DCT
K      = 8
t_vals = arc_length(E)
coeffs, T_curve, N_curve = dct_fit(t_vals, E, K)
restored = dct_eval_batch(t_vals, coeffs, T_curve, N_curve, K)

# ───────────────────────────────────────────────
# 方法A：线性扫描（原来的方法）
# ───────────────────────────────────────────────

def linear_scan(q_emb, top_k=2):
    sims = cosine_similarity(q_emb.reshape(1,-1), restored)[0]
    idx  = np.argsort(sims)[::-1][:top_k]
    return idx.tolist(), sims[idx].tolist()

# ───────────────────────────────────────────────
# 方法B：曲线检索（你的想法）
# ───────────────────────────────────────────────

def find_t_star(q_emb):
    """
    在 DCT 曲线上找离 query 最近的 t*
    distance(t) = ||curve(t) - q||²
    用 scipy 最小化这个1D函数
    """
    def dist(t):
        pt = dct_eval(t, coeffs, T_curve, N_curve, K)
        return np.sum((pt - q_emb) ** 2)

    result = minimize_scalar(dist, bounds=(0, T_curve), method='bounded')
    return result.x

def curve_retrieval(q_emb, window=1.8, top_k=2):
    """原始曲线检索（单一 t*）"""
    t_star     = find_t_star(q_emb)
    candidates = [i for i, t in enumerate(t_vals) if abs(t - t_star) < window]
    if not candidates:
        candidates = [np.argmin(np.abs(t_vals - t_star))]
    cand_embs  = restored[candidates]
    sims       = cosine_similarity(q_emb.reshape(1,-1), cand_embs)[0]
    top        = np.argsort(sims)[::-1][:top_k]
    idx        = [candidates[i] for i in top]
    return idx, sims[top].tolist(), t_star, candidates

# ── 修法1：多起点搜索 ─────────────────────────────

def multistart_retrieval(q_emb, n_starts=6, window=1.8, top_k=2):
    """
    在曲线上均匀取 n_starts 个起点
    各自找局部 t*，取所有候选点的并集
    """
    def dist(t):
        pt = dct_eval(t, coeffs, T_curve, N_curve, K)
        return np.sum((pt - q_emb) ** 2)

    starts     = np.linspace(0, T_curve, n_starts + 2)[1:-1]
    all_t_star = []
    for s in starts:
        res = minimize_scalar(dist, bounds=(max(0, s - T_curve/n_starts),
                                            min(T_curve, s + T_curve/n_starts)),
                              method='bounded')
        all_t_star.append(res.x)

    # 收集所有 t* 窗口内的候选点（去重）
    candidate_set = set()
    for t_star in all_t_star:
        for i, t in enumerate(t_vals):
            if abs(t - t_star) < window:
                candidate_set.add(i)

    candidates = sorted(candidate_set)
    if not candidates:
        candidates = [np.argmin([min(abs(t - ts) for ts in all_t_star)
                                 for t in t_vals])]

    cand_embs = restored[candidates]
    sims      = cosine_similarity(q_emb.reshape(1,-1), cand_embs)[0]
    top       = np.argsort(sims)[::-1][:top_k]
    idx       = [candidates[i] for i in top]
    return idx, sims[top].tolist(), candidates

# ── 修法3：二分搜索替代线性扫描 ──────────────────────

def multistart_binary(q_emb, n_starts=6, window=1.8, top_k=2):
    """
    多起点 + 二分搜索
    t_vals 已排序 → searchsorted 找边界 → O(log N) 替代 O(N) 循环
    """
    def dist(t):
        pt = dct_eval(t, coeffs, T_curve, N_curve, K)
        return np.sum((pt - q_emb) ** 2)

    starts     = np.linspace(0, T_curve, n_starts + 2)[1:-1]
    all_t_star = []
    for s in starts:
        res = minimize_scalar(dist, bounds=(max(0, s - T_curve/n_starts),
                                            min(T_curve, s + T_curve/n_starts)),
                              method='bounded')
        all_t_star.append(res.x)

    # 二分搜索：每个 t* 的 [t*-window, t*+window] 区间边界
    candidate_set = set()
    for ts in all_t_star:
        lo = np.searchsorted(t_vals, ts - window, side='left')
        hi = np.searchsorted(t_vals, ts + window, side='right')
        candidate_set.update(range(lo, hi))

    candidates = sorted(candidate_set)
    if not candidates:
        candidates = [np.argmin([min(abs(t - ts) for ts in all_t_star)
                                 for t in t_vals])]

    cand_embs = restored[candidates]
    sims      = cosine_similarity(q_emb.reshape(1,-1), cand_embs)[0]
    top       = np.argsort(sims)[::-1][:top_k]
    idx       = [candidates[i] for i in top]
    return idx, sims[top].tolist(), candidates

# ── 修法2：t* + 低置信度时退回线性扫描 ──────────────

def adaptive_retrieval(q_emb, window=1.8, threshold=0.35, top_k=2):
    """
    先用曲线检索
    如果最高相似度 < threshold → t* 可能找错了 → 退回线性扫描
    """
    t_star     = find_t_star(q_emb)
    candidates = [i for i, t in enumerate(t_vals) if abs(t - t_star) < window]
    if not candidates:
        candidates = [np.argmin(np.abs(t_vals - t_star))]

    cand_embs  = restored[candidates]
    sims       = cosine_similarity(q_emb.reshape(1,-1), cand_embs)[0]
    best_sim   = sims.max()

    if best_sim >= threshold:
        # 置信度够，用曲线结果
        top = np.argsort(sims)[::-1][:top_k]
        idx = [candidates[i] for i in top]
        return idx, sims[top].tolist(), candidates, "curve"
    else:
        # 置信度低，退回线性扫描
        all_sims = cosine_similarity(q_emb.reshape(1,-1), restored)[0]
        top      = np.argsort(all_sims)[::-1][:top_k]
        return top.tolist(), all_sims[top].tolist(), list(range(N)), "fallback"

# ───────────────────────────────────────────────
# 四种方法对比
# ───────────────────────────────────────────────

print("=" * 80)
print("  五种方法对比")
print("=" * 80)
print(f"\n  {'查询':24}  {'线性':^6}  {'原始曲线':^8}  {'多起点':^8}  {'自适应':^8}  {'二分':^6}")
print(f"  {'─'*72}")

stats = {"linear":    {"hits":0,"checked":[],"times":[]},
         "curve":     {"hits":0,"checked":[],"times":[]},
         "multi":     {"hits":0,"checked":[],"times":[]},
         "adaptive":  {"hits":0,"checked":[],"times":[]},
         "binary":    {"hits":0,"checked":[],"times":[]}}

for query_text, expected in QUERIES:
    q_emb = vec.transform([query_text]).toarray()[0]

    # 线性扫描
    t0 = time.perf_counter()
    l_idx, _ = linear_scan(q_emb)
    stats["linear"]["times"].append((time.perf_counter()-t0)*1000)
    stats["linear"]["checked"].append(N)
    lh = any(i in expected for i in l_idx); stats["linear"]["hits"] += lh

    # 原始曲线
    t0 = time.perf_counter()
    c_idx, _, _, cands = curve_retrieval(q_emb)
    stats["curve"]["times"].append((time.perf_counter()-t0)*1000)
    stats["curve"]["checked"].append(len(cands))
    ch = any(i in expected for i in c_idx); stats["curve"]["hits"] += ch

    # 多起点
    t0 = time.perf_counter()
    m_idx, _, mcands = multistart_retrieval(q_emb)
    stats["multi"]["times"].append((time.perf_counter()-t0)*1000)
    stats["multi"]["checked"].append(len(mcands))
    mh = any(i in expected for i in m_idx); stats["multi"]["hits"] += mh

    # 自适应
    t0 = time.perf_counter()
    a_idx, _, acands, mode = adaptive_retrieval(q_emb)
    stats["adaptive"]["times"].append((time.perf_counter()-t0)*1000)
    stats["adaptive"]["checked"].append(len(acands))
    ah = any(i in expected for i in a_idx); stats["adaptive"]["hits"] += ah

    # 二分搜索
    t0 = time.perf_counter()
    b_idx, _, bcands = multistart_binary(q_emb)
    stats["binary"]["times"].append((time.perf_counter()-t0)*1000)
    stats["binary"]["checked"].append(len(bcands))
    bh = any(i in expected for i in b_idx); stats["binary"]["hits"] += bh

    lm = "✓" if lh else "✗"
    cm = "✓" if ch else "✗"
    mm = "✓" if mh else "✗"
    am = "✓" if ah else "✗"
    bm = "✓" if bh else "✗"
    fb = "(fb)" if mode=="fallback" else "    "
    print(f"  {query_text:24}  {lm:^6}  {cm:^8}  {mm:^8}  {am}{fb}  {bm:^6}")

Q = len(QUERIES)
print(f"\n{'=' * 80}")
print(f"  {'':24}  {'线性':>8}  {'原始曲线':>8}  {'多起点':>8}  {'自适应':>8}  {'二分':>6}")
print(f"  {'─'*72}")
print(f"  {'命中率':24}  {stats['linear']['hits']/Q*100:>7.1f}%  {stats['curve']['hits']/Q*100:>7.1f}%  {stats['multi']['hits']/Q*100:>7.1f}%  {stats['adaptive']['hits']/Q*100:>7.1f}%  {stats['binary']['hits']/Q*100:>5.1f}%")
print(f"  {'平均检查点数':24}  {np.mean(stats['linear']['checked']):>8.1f}  {np.mean(stats['curve']['checked']):>8.1f}  {np.mean(stats['multi']['checked']):>8.1f}  {np.mean(stats['adaptive']['checked']):>8.1f}  {np.mean(stats['binary']['checked']):>6.1f}")
print(f"  {'vs线性减少':24}  {'─':>8}  {(1-np.mean(stats['curve']['checked'])/N)*100:>7.1f}%  {(1-np.mean(stats['multi']['checked'])/N)*100:>7.1f}%  {(1-np.mean(stats['adaptive']['checked'])/N)*100:>7.1f}%  {(1-np.mean(stats['binary']['checked'])/N)*100:>5.1f}%")
print(f"  {'平均延迟':24}  {np.mean(stats['linear']['times']):>7.3f}ms  {np.mean(stats['curve']['times']):>7.3f}ms  {np.mean(stats['multi']['times']):>7.3f}ms  {np.mean(stats['adaptive']['times']):>7.3f}ms  {np.mean(stats['binary']['times']):>5.3f}ms")
print(f"  {'复杂度':24}  {'O(N)':>8}  {'O(1)':>8}  {'O(N)':>8}  {'O(N)':>8}  {'O(logN)':>7}")
print(f"{'=' * 80}")

# 找最优
best = max(["curve","multi","adaptive","binary"],
           key=lambda k: (stats[k]["hits"], -np.mean(stats[k]["checked"])))
print(f"\n  最优方案：{best}")
print(f"  命中率 {stats[best]['hits']}/{Q}，平均检查 {np.mean(stats[best]['checked']):.1f} 个点")
print(f"\n  【二分 vs 多起点对比】")
print(f"  相同命中率：{'是' if stats['binary']['hits'] == stats['multi']['hits'] else '否'}")
print(f"  候选点数：二分 {np.mean(stats['binary']['checked']):.1f} vs 多起点 {np.mean(stats['multi']['checked']):.1f}")
print(f"  延迟：二分 {np.mean(stats['binary']['times']):.3f}ms vs 多起点 {np.mean(stats['multi']['times']):.3f}ms")
print(f"  候选点查找复杂度：O(log N) vs O(N)  （N 越大差距越明显）")
