"""
500字文章 → 两条公式 → 还原文章
诚实测试：字符级 vs 语义级，看看各自能做到什么程度
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────────────────────
# 500字文章（关于AI记忆）
# ─────────────────────────────────────────────────────────────

ARTICLE = """
人工智能的记忆问题是当前研究的核心挑战之一。
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
同时保留关键经历，在有限的存储空间内最大化记忆的价值与效用。
""".strip()

# 提取纯文字（去掉换行）
PURE_TEXT = ARTICLE.replace('\n', '')
print(f"文章总字数: {len(PURE_TEXT)} 字")
print(f"前50字: {PURE_TEXT[:50]}...")

# ─────────────────────────────────────────────────────────────
# 方法一：字符级（Unicode码值序列）
# ─────────────────────────────────────────────────────────────

def fourier_fit_1d(x, y, K):
    T = x[-1]
    Phi = np.zeros((len(x), 2*K+1))
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        Phi[:,2*k-1] = np.cos(2*np.pi*k*x/T)
        Phi[:,2*k  ] = np.sin(2*np.pi*k*x/T)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
    return coeffs, T

def fourier_eval_1d(x, coeffs, T, K):
    Phi = np.zeros((len(x), 2*K+1))
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        Phi[:,2*k-1] = np.cos(2*np.pi*k*x/T)
        Phi[:,2*k  ] = np.sin(2*np.pi*k*x/T)
    return Phi @ coeffs

print("\n" + "="*55)
print("  方法一：字符级压缩（Unicode码值）")
print("="*55)

chars  = list(PURE_TEXT)
N      = len(chars)
codes  = np.array([ord(c) for c in chars], dtype=float)
x_axis = np.arange(N, dtype=float)

print(f"\n原始存储: {N} 个整数")
print(f"{'K':>6}  {'存储':>6}  {'压缩率':>8}  {'字符准确率':>10}  还原前30字")
print("─"*70)

for K in [10, 50, 100, 200]:
    coeffs, T = fourier_fit_1d(x_axis, codes, K)
    recon_codes = fourier_eval_1d(x_axis, coeffs, T, K)
    recon_ints  = np.round(recon_codes).astype(int)
    recon_chars = []
    for code in recon_ints:
        try:    recon_chars.append(chr(max(0, min(0x10FFFF, int(code)))))
        except: recon_chars.append('?')

    accuracy = sum(a==b for a,b in zip(chars, recon_chars)) / N
    storage  = 2*K+1
    compress = (1 - storage/N)*100
    preview  = ''.join(recon_chars[:30])
    print(f"{K:>6}  {storage:>6}  {compress:>7.1f}%  {accuracy:>9.1%}  {preview}")

# ─────────────────────────────────────────────────────────────
# 方法二：句子级语义压缩
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  方法二：句子级语义压缩（TF-IDF embedding）")
print("="*55)

sentences = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
print(f"\n文章共 {len(sentences)} 句，{len(PURE_TEXT)} 字")

# TF-IDF 向量化（字符级）
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
emb_matrix = vectorizer.fit_transform(sentences).toarray()
print(f"每句 embedding 维度: {emb_matrix.shape[1]}")

def arc_length(embs):
    t = [0.0]
    for i in range(1, len(embs)):
        t.append(t[-1] + np.linalg.norm(embs[i] - embs[i-1]))
    return np.array(t)

def fourier_fit_nd(t, embs, K):
    T   = t[-1]
    n,d = embs.shape
    Phi = np.zeros((n, 2*K+1))
    Phi[:,0] = 1.0
    for k in range(1, K+1):
        Phi[:,2*k-1] = np.cos(2*np.pi*k*t/T)
        Phi[:,2*k  ] = np.sin(2*np.pi*k*t/T)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, embs, rcond=None)
    return coeffs, T

def fourier_eval_nd(t_query, coeffs, T, K):
    phi = np.zeros(2*K+1)
    phi[0] = 1.0
    for k in range(1, K+1):
        phi[2*k-1] = np.cos(2*np.pi*k*t_query/T)
        phi[2*k  ] = np.sin(2*np.pi*k*t_query/T)
    return phi @ coeffs

t_vals = arc_length(emb_matrix)
d      = emb_matrix.shape[1]
n_sent = len(sentences)

print(f"\n{'K':>4}  {'公式大小':>8}  {'原始大小':>8}  {'压缩率':>8}  {'句子准确率':>10}")
print("─"*55)

for K in [2, 4, 6, 8]:
    coeffs, T = fourier_fit_nd(t_vals, emb_matrix, K)

    correct = 0
    for i, sent in enumerate(sentences):
        restored_emb = fourier_eval_nd(t_vals[i], coeffs, T, K)
        sims = [(1 - cosine(restored_emb, emb_matrix[j]), j)
                for j in range(n_sent)]
        best_j = max(sims)[1]
        if best_j == i:
            correct += 1

    formula_size = (2*K+1) * d
    original     = n_sent * d
    compress     = (1 - formula_size/original)*100
    accuracy     = correct/n_sent

    print(f"{K:>4}  {formula_size:>8}  {original:>8}  {compress:>7.1f}%  {accuracy:>9.1%}")

# ─────────────────────────────────────────────────────────────
# 最终：用最佳 K 重建文章
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  用公式重建文章（K=8）")
print("="*55)

K = 8
coeffs, T = fourier_fit_nd(t_vals, emb_matrix, K)

print(f"\n存储的东西:")
print(f"  公式系数矩阵  : ({2*K+1}) × ({d}) = {(2*K+1)*d} 个数字")
print(f"  t 值          : {n_sent} 个数字")
print(f"  原始大小      : {n_sent * d} 个数字")
print(f"  压缩率        : {(1-(2*K+1)*d/(n_sent*d))*100:.1f}%")

print(f"\n从公式重建文章:")
print("─"*55)
for i in range(n_sent):
    restored_emb = fourier_eval_nd(t_vals[i], coeffs, T, K)
    sims = [(1 - cosine(restored_emb, emb_matrix[j]), j)
            for j in range(n_sent)]
    best_sim, best_j = max(sims)
    match = "✓" if best_j == i else "✗"
    print(f"  {match} [{best_sim:.3f}] {sentences[best_j][:40]}...")
