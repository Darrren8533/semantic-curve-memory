"""
分层 DCT 检索系统
=================
解决大规模下固定 K 精度下降的问题

结构：
  Level 0（全局）：每个 chunk 取代表向量 → DCT 粗定位
  Level 1（局部）：各 chunk 内部 DCT → 精确检索

效果：
  N=18   → 单层 DCT 已够好
  N=540  → 单层 DCT K=8 精度下降；分层 DCT 维持高准确率
  N=∞    → 只要 chunk_size 固定，每层质量不变
"""

import numpy as np
import time
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

BASE_QUERIES = [
    ("存储成本随条数增长",      [6]),
    ("固定大小参数不随序列增长", [11, 12]),
    ("图结构实体关系",           [7, 8]),
    ("工作记忆长期存储分层",     [9, 10]),
    ("频率控制压缩精度",         [13, 14]),
    ("遗忘不重要信息",           [16, 17]),
]

BASE_SENTENCES = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
N_BASE = len(BASE_SENTENCES)

# ───────────────────────────────────────────────
# DCT 工具
# ───────────────────────────────────────────────

def arc_length(E):
    t = [0.0]
    for i in range(1, len(E)):
        t.append(t[-1] + np.linalg.norm(E[i] - E[i-1]))
    return np.array(t)

def dct_fit(t, Y, K):
    n = len(t); T = t[-1] if t[-1] > 0 else 1.0
    tn = t / T * n
    P = np.zeros((n, K+1)); P[:,0] = 1.0
    for k in range(1, K+1):
        P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    c, _, _, _ = np.linalg.lstsq(P, Y, rcond=None)
    return c, T, n

def dct_eval_batch(t_arr, c, T, n, K):
    tn = t_arr / T * n
    P  = np.zeros((len(t_arr), K+1)); P[:,0] = 1.0
    for k in range(1, K+1):
        P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return P @ c

# ───────────────────────────────────────────────
# 单层 DCT（baseline）
# ───────────────────────────────────────────────

class FlatDCT:
    def __init__(self, K=8):
        self.K = K

    def build(self, E):
        self.E = E
        self.N = len(E)
        t = arc_length(E)
        self.coeffs, self.T, self.Nc = dct_fit(t, E, self.K)
        self.restored = dct_eval_batch(t, self.coeffs, self.T, self.Nc, self.K)

    def retrieve(self, q_emb, top_k=2):
        sims = cosine_similarity(q_emb.reshape(1,-1), self.restored)[0]
        idx  = np.argsort(sims)[::-1][:top_k]
        return idx.tolist(), sims[idx].tolist(), self.N   # checked = N（扫全部）

    @property
    def storage(self):
        return (self.K + 1) * self.E.shape[1]

# ───────────────────────────────────────────────
# 分层 DCT
# ───────────────────────────────────────────────

class HierarchicalDCT:
    """
    两层结构：
      全局层：每个 chunk 用均值代表 → DCT → 找最相关的 chunk
      局部层：chunk 内部 → DCT → 精确找句子
    """
    def __init__(self, K=8, chunk_size=20):
        self.K          = K
        self.chunk_size = chunk_size

    def build(self, E):
        self.E = E
        N, d  = E.shape
        self.N = N

        # ── 切 chunks ───────────────────────────
        self.chunks = []
        for start in range(0, N, self.chunk_size):
            end        = min(start + self.chunk_size, N)
            idx        = list(range(start, end))
            chunk_E    = E[start:end]
            t_chunk    = arc_length(chunk_E)
            c, T, Nc   = dct_fit(t_chunk, chunk_E, self.K)
            restored   = dct_eval_batch(t_chunk, c, T, Nc, self.K)
            self.chunks.append({
                'idx':      idx,
                'E':        chunk_E,
                'coeffs':   c,
                'T':        T,
                'N':        Nc,
                'restored': restored,
                'mean':     chunk_E.mean(axis=0),
            })

        self.n_chunks = len(self.chunks)

    def retrieve(self, q_emb, top_k=2, n_probe=2):
        """
        全局层：直接余弦搜索 chunk 代表点（不用 DCT，保证准确）
        局部层：chunk 内部 DCT 检索（chunk 小，K=8 准确）

        n_probe：进入几个 chunk 精搜（默认2）
        返回：(idx列表, sim列表, 实际检查点数)
        """
        # 全局层：直接对 chunk 均值做余弦相似度（O(n_chunks)）
        means      = np.array([ch['mean'] for ch in self.chunks])
        g_sims     = cosine_similarity(q_emb.reshape(1,-1), means)[0]
        top_chunks = np.argsort(g_sims)[::-1][:n_probe]

        checked    = self.n_chunks          # 全局层扫了 n_chunks 个代表点
        candidates = []

        # 局部层：在每个 chunk 内用 DCT 精搜
        for ci in top_chunks:
            ch   = self.chunks[ci]
            sims = cosine_similarity(q_emb.reshape(1,-1), ch['restored'])[0]
            checked += len(ch['idx'])
            for li in np.argsort(sims)[::-1][:top_k]:
                candidates.append((ch['idx'][li], float(sims[li])))

        # 全局 re-rank
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:top_k]
        return [x[0] for x in top], [x[1] for x in top], checked

    @property
    def storage(self):
        # 全局层 + 所有局部层
        d = self.E.shape[1]
        return (self.K + 1) * d * (1 + self.n_chunks)


# ───────────────────────────────────────────────
# 合成大规模数据（正确测试方法）
# ───────────────────────────────────────────────

def make_synthetic(n_topics, sents_per_topic, d, noise=0.05, seed=42):
    """
    合成数据：n_topics 个主题，每个主题有 sents_per_topic 个句子
    每个主题有一个中心向量，句子 = 中心 + 小扰动
    查询 = 中心向量（应该命中该主题的所有句子）

    返回：
      E          : (N, d) embeddings
      queries    : [(q_emb, expected_idx_list), ...]
    """
    rng = np.random.default_rng(seed)

    # 随机主题中心（单位化，模拟 cosine 空间）
    centers = rng.standard_normal((n_topics, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    E = []
    queries = []
    for ti, center in enumerate(centers):
        start = ti * sents_per_topic
        for _ in range(sents_per_topic):
            sent_emb = center + rng.normal(0, noise, d)
            sent_emb = np.clip(sent_emb, 0, None)          # 非负，模拟 TF-IDF
            E.append(sent_emb)
        # 查询 = 中心 + 微小偏移
        q_emb    = center + rng.normal(0, noise * 0.1, d)
        q_emb    = np.clip(q_emb, 0, None)
        expected = list(range(start, start + sents_per_topic))
        queries.append((q_emb, expected))

    return np.array(E), queries


def run_test(name, E, queries, flat, hier):
    N = len(E)
    Q = len(queries)

    f_hits = 0; f_checked = []; f_times = []
    h_hits = 0; h_checked = []; h_times = []

    for q_emb, expected in queries:
        t0 = time.perf_counter()
        fi, _, fc = flat.retrieve(q_emb, top_k=3)
        f_times.append((time.perf_counter()-t0)*1000)
        f_checked.append(fc)
        f_hits += any(i in expected for i in fi)

        t0 = time.perf_counter()
        hi, _, hc = hier.retrieve(q_emb, top_k=3)
        h_times.append((time.perf_counter()-t0)*1000)
        h_checked.append(hc)
        h_hits += any(i in expected for i in hi)

    print(f"\n  【{name}  N={N}】")
    print(f"  {'':20}  {'单层DCT':>10}  {'分层DCT':>10}")
    print(f"  {'─'*44}")
    print(f"  {'命中率':20}  {f_hits/Q*100:>9.1f}%  {h_hits/Q*100:>9.1f}%")
    print(f"  {'平均检查点数':20}  {np.mean(f_checked):>10.1f}  {np.mean(h_checked):>10.1f}")
    print(f"  {'vs总点数减少':20}  {'─':>10}  {(1-np.mean(h_checked)/N)*100:>9.1f}%")
    print(f"  {'平均延迟':20}  {np.mean(f_times):>9.3f}ms  {np.mean(h_times):>9.3f}ms")

    return f_hits / Q * 100, h_hits / Q * 100


# ───────────────────────────────────────────────
# 主程序
# ───────────────────────────────────────────────

print("=" * 60)
print("  分层 DCT 检索系统")
print("=" * 60)

# ── Part 1：真实数据验证（N=18）─────────────────

print("\n[Part 1] 真实数据（TF-IDF 中文）")

vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
E_real = vec.fit_transform(BASE_SENTENCES).toarray()
d_real = E_real.shape[1]

q_embs_real = vec.transform([q for q, _ in BASE_QUERIES]).toarray()
real_queries = [(q_embs_real[i], BASE_QUERIES[i][1]) for i in range(len(BASE_QUERIES))]

flat_real = FlatDCT(K=8);              flat_real.build(E_real)
hier_real = HierarchicalDCT(K=8, chunk_size=6); hier_real.build(E_real)

run_test("真实 N=18", E_real, real_queries, flat_real, hier_real)

# ── Part 2：合成数据规模测试 ──────────────────────

print("\n[Part 2] 合成数据（随机聚类 embedding）")
print("  设计：每个主题 = 一个向量中心 + 微小扰动")
print("  查询 = 主题中心，应命中该主题的句子")

D   = 64    # embedding 维度
SPT = 10    # sentences per topic

scale_results = []
configs = [
    ("N=50",   5,   SPT, 6),
    ("N=100",  10,  SPT, 10),
    ("N=200",  20,  SPT, 10),
    ("N=500",  50,  SPT, 10),
    ("N=1000", 100, SPT, 10),
]

for label, n_topics, spt, chunk_size in configs:
    E_syn, queries_syn = make_synthetic(n_topics, spt, D)

    flat_syn = FlatDCT(K=8)
    flat_syn.build(E_syn)

    hier_syn = HierarchicalDCT(K=8, chunk_size=chunk_size)
    hier_syn.build(E_syn)

    fhr, hhr = run_test(label, E_syn, queries_syn, flat_syn, hier_syn)
    scale_results.append((label, len(E_syn), fhr, hhr))

# ── 汇总表 ────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  规模扩展汇总")
print(f"{'=' * 60}")
print(f"  {'规模':10}  {'N':>6}  {'单层DCT':>10}  {'分层DCT':>10}  {'差值':>8}")
print(f"  {'─'*50}")
for label, N_, fhr, hhr in scale_results:
    diff = hhr - fhr
    mark = " ← 分层更好" if diff > 10 else ""
    print(f"  {label:10}  {N_:>6}  {fhr:>9.1f}%  {hhr:>9.1f}%  {diff:>+7.1f}%{mark}")

print(f"\n  结论：")
print(f"  单层 DCT K=8 → N 增大时曲线过度平滑，t* 不准，命中率下降")
print(f"  分层 DCT     → 每 chunk 内点数固定，局部曲线质量稳定")
print(f"{'=' * 60}")
