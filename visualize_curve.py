"""
语义曲线可视化
==============
把高维 embedding 降到 2D，画出：
  - 原始记忆点（散点）
  - DCT 拟合曲线（连续线）
  - 查询点 + 检索路径

直观展示：两条线的交叉点 = 记忆
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

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
    ("存储成本随条数增长",      [6]),
    ("固定大小参数不随序列增长", [11, 12]),
    ("图结构实体关系",           [7, 8]),
    ("工作记忆长期存储分层",     [9, 10]),
    ("频率控制压缩精度",         [13, 14]),
    ("遗忘不重要信息",           [16, 17]),
]

sentences = [s.strip() for s in ARTICLE.strip().split('\n') if s.strip()]
N = len(sentences)

# ───────────────────────────────────────────────
# Embedding + PCA 降维
# ───────────────────────────────────────────────

vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
E   = vec.fit_transform(sentences).toarray()

pca  = PCA(n_components=2)
E2   = pca.fit_transform(E)          # (18, 2) — 原始记忆点在2D中的位置

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

def dct_eval(tq, c, T, n, K):
    tn = tq / T * n; phi = np.zeros(K+1); phi[0] = 1.0
    for k in range(1, K+1): phi[k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return phi @ c

def dct_eval_batch(t_arr, c, T, n, K):
    tn = t_arr / T * n
    P  = np.zeros((len(t_arr), K+1)); P[:,0] = 1.0
    for k in range(1, K+1): P[:,k] = np.cos(np.pi / n * (tn + 0.5) * k)
    return P @ c

# 高维 DCT 拟合
K = 8
t_vals = arc_length(E)
coeffs, T_c, N_c = dct_fit(t_vals, E, K)

# 在高维空间采样曲线，再投影到 2D
t_dense   = np.linspace(0, T_c, 300)
curve_hd  = dct_eval_batch(t_dense, coeffs, T_c, N_c, K)   # (300, d)
curve_2d  = pca.transform(curve_hd)                          # (300, 2)

# 还原的记忆点（曲线上对应 t_vals 的位置）
restored_hd = dct_eval_batch(t_vals, coeffs, T_c, N_c, K)
restored_2d = pca.transform(restored_hd)

# ───────────────────────────────────────────────
# 选两个示例查询
# ───────────────────────────────────────────────

DEMO_QUERIES = [
    ("存储成本随条数增长",      [6]),
    ("工作记忆长期存储分层",    [9, 10]),
]

q_embs_hd = vec.transform([q for q, _ in DEMO_QUERIES]).toarray()
q_embs_2d = pca.transform(q_embs_hd)

# 多起点检索：找 t* 列表
def find_t_stars(q_emb_hd, n_starts=6):
    def dist(t):
        return np.sum((dct_eval(t, coeffs, T_c, N_c, K) - q_emb_hd) ** 2)
    starts = np.linspace(0, T_c, n_starts + 2)[1:-1]
    t_stars = []
    for s in starts:
        w = T_c / n_starts
        res = minimize_scalar(dist, bounds=(max(0, s-w), min(T_c, s+w)),
                              method='bounded')
        t_stars.append(res.x)
    return t_stars

# ───────────────────────────────────────────────
# 图1：总览 — 语义曲线全貌
# ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('DCT 语义曲线记忆系统 — 可视化（PCA 2D投影）', fontsize=14, fontweight='bold', y=1.01)

COLORS = {
    '向量数据库': '#e74c3c',
    '知识图谱':   '#e67e22',
    '分层记忆':   '#2ecc71',
    '状态空间':   '#3498db',
    '傅里叶':     '#9b59b6',
    '其他':       '#95a5a6',
}

# 按主题给句子分组
GROUPS = {
    '向量数据库': [4, 5, 6],
    '知识图谱':   [7, 8],
    '分层记忆':   [9, 10],
    '状态空间':   [11, 12],
    '傅里叶':     [13, 14],
    '其他':       [0, 1, 2, 3, 15, 16, 17],
}

def get_color(idx):
    for grp, idxs in GROUPS.items():
        if idx in idxs:
            return COLORS[grp]
    return COLORS['其他']

# ── 左图：原始记忆点 vs DCT 曲线 ──────────────────

ax = axes[0]
ax.set_title('线1（DCT曲线）vs 线2（原始记忆点）', fontsize=12)

# DCT 曲线（线1）
ax.plot(curve_2d[:, 0], curve_2d[:, 1],
        color='#2980b9', lw=2.5, alpha=0.6, zorder=1, label='DCT 曲线（公式线）')

# 曲线方向箭头
for frac in [0.2, 0.5, 0.8]:
    i = int(frac * len(curve_2d))
    dx = curve_2d[i+1, 0] - curve_2d[i-1, 0]
    dy = curve_2d[i+1, 1] - curve_2d[i-1, 1]
    ax.annotate('', xy=(curve_2d[i, 0]+dx*0.01, curve_2d[i, 1]+dy*0.01),
                xytext=(curve_2d[i, 0], curve_2d[i, 1]),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

# 原始记忆点（线2 = 向量空间中的点）
for i, (x, y) in enumerate(E2):
    c = get_color(i)
    ax.scatter(x, y, s=120, color=c, zorder=4, edgecolors='white', linewidths=1.2)

# 交叉点（DCT曲线上对应位置）= 两线的交叉 = 真正的记忆
for i, (x, y) in enumerate(restored_2d):
    c = get_color(i)
    ax.scatter(x, y, s=60, marker='*', color=c, zorder=5, alpha=0.7)
    # 连线：原始点 → 曲线上的还原点
    ax.plot([E2[i, 0], x], [E2[i, 1], y],
            color=c, lw=0.8, alpha=0.35, linestyle='--', zorder=2)

# 标注句子序号
for i, (x, y) in enumerate(E2):
    ax.annotate(str(i+1), (x, y), textcoords='offset points', xytext=(5, 5),
                fontsize=7.5, color='#2c3e50')

# 图例：主题色
patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()]
ax.legend(handles=patches, fontsize=8, loc='upper right',
          title='主题分组', title_fontsize=8)
ax.set_xlabel('PCA 维度 1', fontsize=9)
ax.set_ylabel('PCA 维度 2', fontsize=9)
ax.grid(True, alpha=0.2)

# ── 右图：查询检索过程 ────────────────────────────

ax = axes[1]
ax.set_title('查询检索：新线 → 找交叉点', fontsize=12)

# 背景：DCT 曲线
ax.plot(curve_2d[:, 0], curve_2d[:, 1],
        color='#2980b9', lw=2.0, alpha=0.4, zorder=1, label='DCT 曲线')

# 原始记忆点（灰色底层）
for i, (x, y) in enumerate(E2):
    ax.scatter(x, y, s=80, color='#bdc3c7', zorder=3, edgecolors='white', linewidths=1)
    ax.annotate(str(i+1), (x, y), textcoords='offset points', xytext=(4, 4),
                fontsize=7, color='#7f8c8d')

Q_COLORS = ['#e74c3c', '#2ecc71']

for qi, ((q_text, expected), q_emb_hd, q_pt_2d) in enumerate(
        zip(DEMO_QUERIES, q_embs_hd, q_embs_2d)):

    qc = Q_COLORS[qi]

    # 查询点
    ax.scatter(*q_pt_2d, s=200, color=qc, marker='D', zorder=7,
               edgecolors='white', linewidths=1.5)
    ax.annotate(f'Q{qi+1}: {q_text[:8]}…', q_pt_2d,
                textcoords='offset points', xytext=(8, 6),
                fontsize=8, color=qc, fontweight='bold')

    # 多起点找 t*
    t_stars = find_t_stars(q_emb_hd)
    window  = T_c * 0.15

    # 在曲线上标出 t* 位置
    for ts in t_stars:
        pt_hd = dct_eval(ts, coeffs, T_c, N_c, K).reshape(1, -1)
        pt_2d = pca.transform(pt_hd)[0]
        ax.scatter(*pt_2d, s=50, color=qc, marker='x', linewidths=1.5,
                   zorder=6, alpha=0.5)

    # 候选点（窗口内）
    t_lo = np.searchsorted(t_vals, min(t_stars) - window)
    t_hi = np.searchsorted(t_vals, max(t_stars) + window)
    cand_idxs = list(range(max(0, t_lo), min(N, t_hi+1)))

    for ci in cand_idxs:
        ax.scatter(*E2[ci], s=130, color=qc, zorder=5,
                   edgecolors=qc, linewidths=1.5, alpha=0.4)

    # 命中点（expected）
    for ei in expected:
        ax.scatter(*E2[ei], s=200, color=qc, marker='*', zorder=8,
                   edgecolors='white', linewidths=1)
        # 查询点 → 命中点的箭头
        ax.annotate('', xy=E2[ei], xytext=q_pt_2d,
                    arrowprops=dict(arrowstyle='->', color=qc,
                                   lw=1.5, connectionstyle='arc3,rad=0.25'))

# 图例
leg_items = [
    plt.scatter([], [], s=150, color='#e74c3c', marker='D', label='Q1 查询点'),
    plt.scatter([], [], s=150, color='#2ecc71', marker='D', label='Q2 查询点'),
    plt.scatter([], [], s=150, color='#e74c3c', marker='*', label='Q1 命中点'),
    plt.scatter([], [], s=150, color='#2ecc71', marker='*', label='Q2 命中点'),
    plt.scatter([], [], s=80,  color='#bdc3c7', label='记忆点'),
]
ax.legend(handles=leg_items, fontsize=8, loc='upper right')
ax.set_xlabel('PCA 维度 1', fontsize=9)
ax.set_ylabel('PCA 维度 2', fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('semantic_curve_viz.png', dpi=150, bbox_inches='tight')
print("图1 已保存: semantic_curve_viz.png")
plt.close()

# ───────────────────────────────────────────────
# 图2：压缩率 vs 命中率（不同 K 值）
# ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('DCT K 值：压缩率 vs 命中率', fontsize=12, fontweight='bold')

K_vals      = [2, 4, 6, 8, 10, 12]
d           = E.shape[1]
hit_rates   = []
compress    = []

QUERIES_ALL = [
    ("存储成本随条数增长",      [6]),
    ("固定大小参数不随序列增长", [11, 12]),
    ("图结构实体关系",           [7, 8]),
    ("工作记忆长期存储分层",     [9, 10]),
    ("频率控制压缩精度",         [13, 14]),
    ("遗忘不重要信息",           [16, 17]),
]
q_embs_all = vec.transform([q for q, _ in QUERIES_ALL]).toarray()

for Kv in K_vals:
    c_, T_, N_ = dct_fit(t_vals, E, Kv)
    r_  = dct_eval_batch(t_vals, c_, T_, N_, Kv)
    hits = 0
    for (_, exp), qe in zip(QUERIES_ALL, q_embs_all):
        sims = cosine_similarity(qe.reshape(1,-1), r_)[0]
        top2 = np.argsort(sims)[::-1][:2]
        if any(i in exp for i in top2): hits += 1
    hit_rates.append(hits / len(QUERIES_ALL) * 100)
    compress.append((1 - (Kv+1)*d / (N*d)) * 100)

color_pts = ['#e74c3c' if h < 100 else '#2ecc71' for h in hit_rates]

ax2 = ax.twinx()
bars = ax.bar([str(k) for k in K_vals], compress, color='#3498db', alpha=0.4,
              label='压缩率 %')
line, = ax2.plot([str(k) for k in K_vals], hit_rates, 'o-',
                 color='#e74c3c', lw=2, ms=8, label='命中率 %')

for i, (k, c, h) in enumerate(zip(K_vals, compress, hit_rates)):
    ax.text(i, c + 1, f'{c:.0f}%', ha='center', fontsize=9, color='#2980b9')
    ax2.text(i, h + 1.5, f'{h:.0f}%', ha='center', fontsize=9, color='#c0392b')

ax.set_xlabel('K（DCT 频率数）', fontsize=10)
ax.set_ylabel('压缩率 %', fontsize=10, color='#2980b9')
ax2.set_ylabel('命中率 %', fontsize=10, color='#e74c3c')
ax2.set_ylim(0, 115)
ax.set_ylim(0, 80)

ax.axvline(x=K_vals.index(8) - 0.4, color='orange', lw=0, alpha=0)
ax.get_children()[K_vals.index(8)].set_facecolor('#e67e22')
ax.get_children()[K_vals.index(8)].set_alpha(0.7)
ax.text(K_vals.index(8), 2, '最优', ha='center', fontsize=9,
        color='white', fontweight='bold')

lines = [bars, line]
labels = ['压缩率 %', '命中率 %']
ax.legend(lines, labels, loc='lower right', fontsize=9)

ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('k_tradeoff.png', dpi=150, bbox_inches='tight')
print("图2 已保存: k_tradeoff.png")
plt.close()

print("\n完成！两张图已保存在当前目录。")
print("  semantic_curve_viz.png — 语义曲线全貌 + 检索路径")
print("  k_tradeoff.png         — K 值：压缩率 vs 命中率")
