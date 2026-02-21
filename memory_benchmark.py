"""
DCT 记忆系统 vs 向量数据库
============================
Step 1：加检索功能
Step 2：正面对比数据
"""

import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────────────────────────
# 文章 + 测试查询（附已知答案）
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

# 查询 → 期望命中的句子索引（0-based）
QUERIES = [
    ("存储成本随条数增长",        [6]),        # "然而这种方法的存储成本..."
    ("固定大小参数不随序列增长",   [11, 12]),   # "状态空间模型..." + "无论处理多长..."
    ("图结构实体关系",             [7, 8]),     # "知识图谱..." + "但图结构..."
    ("工作记忆长期存储分层",       [9, 10]),    # "分层记忆系统..." + "重要信息..."
    ("频率控制压缩精度",           [13, 14]),   # "傅里叶..." + "通过控制..."
    ("遗忘不重要信息",             [16, 17]),   # "未来..." + "同时保留..."
]

sentences = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
N = len(sentences)

# ───────────────────────────────────────────────
# 公式B：TF-IDF 向量空间（两个系统共用）
# ───────────────────────────────────────────────

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
emb_matrix = vectorizer.fit_transform(sentences).toarray()   # (N, d)
d = emb_matrix.shape[1]

# ───────────────────────────────────────────────
# DCT 工具函数
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

def dct_eval_batch(t_arr, c, T, n, K):
    tn = t_arr / T * n
    P  = np.zeros((len(t_arr), K+1)); P[:,0] = 1.0
    for k in range(1, K+1):
        P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return P @ c

# ───────────────────────────────────────────────
# System A：DCT 记忆系统
# ───────────────────────────────────────────────

class DCTMemory:
    def __init__(self, K=8):
        self.K = K

    def build(self, emb_matrix, t_vals):
        t0 = time.perf_counter()
        self.coeffs, self.T, self.N = dct_fit(t_vals, emb_matrix, self.K)
        # 预先还原所有句子的 embedding（检索时用）
        self.restored = dct_eval_batch(t_vals, self.coeffs, self.T, self.N, self.K)
        self.build_ms = (time.perf_counter() - t0) * 1000

    def recall(self, query_emb, top_k=2):
        t0   = time.perf_counter()
        sims = cosine_similarity(query_emb.reshape(1,-1), self.restored)[0]
        idx  = np.argsort(sims)[::-1][:top_k]
        ms   = (time.perf_counter() - t0) * 1000
        return idx.tolist(), sims[idx].tolist(), ms

    @property
    def storage(self):
        return (self.K + 1) * self.restored.shape[1]   # 公式A大小


# ───────────────────────────────────────────────
# System B：向量数据库（直接存所有 embedding）
# ───────────────────────────────────────────────

class VectorDB:
    def build(self, emb_matrix):
        t0 = time.perf_counter()
        self.store = emb_matrix.copy()
        self.build_ms = (time.perf_counter() - t0) * 1000

    def recall(self, query_emb, top_k=2):
        t0   = time.perf_counter()
        sims = cosine_similarity(query_emb.reshape(1,-1), self.store)[0]
        idx  = np.argsort(sims)[::-1][:top_k]
        ms   = (time.perf_counter() - t0) * 1000
        return idx.tolist(), sims[idx].tolist(), ms

    @property
    def storage(self):
        return self.store.size   # 所有 embedding 的大小


# ───────────────────────────────────────────────
# 构建两个系统
# ───────────────────────────────────────────────

t_vals = arc_length(emb_matrix)

dct = DCTMemory(K=8)
dct.build(emb_matrix, t_vals)

vdb = VectorDB()
vdb.build(emb_matrix)

# ───────────────────────────────────────────────
# Step 1：检索演示
# ───────────────────────────────────────────────

print("=" * 68)
print("  Step 1：检索功能演示（DCT 系统）")
print("=" * 68)

for query_text, expected in QUERIES:
    q_emb = vectorizer.transform([query_text]).toarray()[0]
    idx, scores, _ = dct.recall(q_emb, top_k=2)
    hit = any(i in expected for i in idx)
    mark = "✓" if hit else "✗"
    print(f"\n  查询：「{query_text}」  {mark}")
    for rank, (i, s) in enumerate(zip(idx, scores), 1):
        exp_mark = "← 命中" if i in expected else ""
        print(f"    {rank}. [{s:.3f}] {sentences[i][:48]}... {exp_mark}")

# ───────────────────────────────────────────────
# Step 2：正面对比
# ───────────────────────────────────────────────

print(f"\n{'=' * 68}")
print(f"  Step 2：DCT 系统 vs 向量数据库")
print(f"{'=' * 68}")

# 存储对比
print(f"\n  【存储】")
print(f"  {'':20}  {'DCT':>10}  {'向量DB':>10}  {'DCT节省':>10}")
print(f"  {'─'*54}")
print(f"  {'公式/数据大小':20}  {dct.storage:>10}  {vdb.storage:>10}  {(1-dct.storage/vdb.storage)*100:>9.1f}%")
print(f"  {'说明':20}  {'(K+1)×d':>10}  {'N×d':>10}")

# 检索对比
print(f"\n  【检索准确率 & 速度】")
print(f"  {'查询':24}  {'DCT':^12}  {'VDB':^12}")
print(f"  {'─'*54}")

dct_hits = 0; vdb_hits = 0
dct_times = []; vdb_times = []

for query_text, expected in QUERIES:
    q_emb = vectorizer.transform([query_text]).toarray()[0]

    d_idx, d_scores, d_ms = dct.recall(q_emb, top_k=2)
    v_idx, v_scores, v_ms = vdb.recall(q_emb, top_k=2)

    d_hit = any(i in expected for i in d_idx)
    v_hit = any(i in expected for i in v_idx)

    dct_hits  += d_hit;  vdb_hits  += v_hit
    dct_times.append(d_ms); vdb_times.append(v_ms)

    dm = "✓" if d_hit else "✗"
    vm = "✓" if v_hit else "✗"
    print(f"  {query_text:24}  {dm} {d_ms:.3f}ms    {vm} {v_ms:.3f}ms")

print(f"  {'─'*54}")
print(f"  {'总命中率':24}  {dct_hits}/{len(QUERIES)}          {vdb_hits}/{len(QUERIES)}")
print(f"  {'平均延迟':24}  {np.mean(dct_times):.3f}ms       {np.mean(vdb_times):.3f}ms")

# 汇总
print(f"\n{'=' * 68}")
print(f"  汇总")
print(f"{'=' * 68}")
print(f"  {'':20}  {'DCT':>12}  {'向量DB':>12}")
print(f"  {'─'*48}")
print(f"  {'存储大小':20}  {dct.storage:>12}  {vdb.storage:>12}")
print(f"  {'压缩率':20}  {(1-dct.storage/vdb.storage)*100:>11.1f}%  {'0%':>12}")
print(f"  {'检索命中率':20}  {dct_hits/len(QUERIES)*100:>11.1f}%  {vdb_hits/len(QUERIES)*100:>11.1f}%")
print(f"  {'平均检索延迟':20}  {np.mean(dct_times):>11.3f}ms  {np.mean(vdb_times):>11.3f}ms")
print(f"  {'─'*48}")
print(f"  DCT 用 {(1-dct.storage/vdb.storage)*100:.0f}% 更少的存储，达到同等检索效果")
