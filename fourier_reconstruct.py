"""
解决方案：字符共现 embedding + 傅里叶压缩 → 还原文章

思路：
  原问题：Unicode码值信号太跳跃，傅里叶拟合失败
  解法：  用字符在文章里的共现关系建立embedding
         共现越多的字 → embedding越接近 → 信号越平滑 → 傅里叶有效

两条"公式"：
  公式A：傅里叶系数矩阵（压缩文章的语义轨迹）
  公式B：字符embedding表（解码器，可跨文章共用）

两者缺一不可，合起来才能还原原文
"""

import numpy as np
from scipy.spatial.distance import cdist

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
""".strip().replace('\n', '')


# ─────────────────────────────────────────────
# Step 1：用共现关系建立字符 embedding（公式B）
# ─────────────────────────────────────────────

def build_char_embeddings(text, window=4, dim=8):
    """
    共现矩阵 → SVD → 字符embedding
    同一窗口内经常同时出现的字 → embedding接近
    这就是 word2vec 的核心思想
    """
    unique = list(dict.fromkeys(text))   # 保持出现顺序，去重
    idx    = {c: i for i, c in enumerate(unique)}
    n      = len(unique)

    # 构建共现矩阵
    cooc = np.zeros((n, n))
    for pos, char in enumerate(text):
        i = idx[char]
        for delta in range(-window, window + 1):
            if delta != 0 and 0 <= pos + delta < len(text):
                j = idx[text[pos + delta]]
                cooc[i, j] += 1

    # SVD 降维到 dim 维
    U, S, _ = np.linalg.svd(cooc, full_matrices=False)
    embs    = U[:, :dim] * S[:dim]          # shape: (n_unique, dim)

    return unique, idx, embs                # embs = 公式B


# ─────────────────────────────────────────────
# Step 2：傅里叶压缩（公式A）
# ─────────────────────────────────────────────

def fourier_fit(x, Y, K):
    """Y: (N, d)  →  coeffs: (2K+1, d)"""
    T   = float(x[-1])
    N   = len(x)
    Phi = np.zeros((N, 2*K+1))
    Phi[:, 0] = 1.0
    for k in range(1, K+1):
        Phi[:, 2*k-1] = np.cos(2*np.pi*k*x/T)
        Phi[:, 2*k  ] = np.sin(2*np.pi*k*x/T)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    return coeffs, T

def fourier_eval(x_query, coeffs, T, K):
    phi    = np.zeros(2*K+1)
    phi[0] = 1.0
    for k in range(1, K+1):
        phi[2*k-1] = np.cos(2*np.pi*k*x_query/T)
        phi[2*k  ] = np.sin(2*np.pi*k*x_query/T)
    return phi @ coeffs


# ─────────────────────────────────────────────
# Step 3：还原——从 embedding 找回最近的字
# ─────────────────────────────────────────────

def reconstruct(text, char_embs_seq, coeffs, T, K, unique, all_embs):
    """
    对每个位置：
      公式A(t) → 还原embedding → 在公式B里找最近的字
    """
    N       = len(text)
    x_axis  = np.arange(N, dtype=float)
    result  = []

    # 批量还原所有位置的 embedding
    restored = np.array([fourier_eval(x, coeffs, T, K) for x in x_axis])

    # 批量计算距离（向量化，快）
    dists    = cdist(restored, all_embs, metric='euclidean')  # (N, n_unique)
    best_idx = np.argmin(dists, axis=1)
    result   = [unique[i] for i in best_idx]

    accuracy = sum(a == b for a, b in zip(text, result)) / N
    return ''.join(result), accuracy


# ─────────────────────────────────────────────
# 运行
# ─────────────────────────────────────────────

text = ARTICLE
N    = len(text)
print(f"原文长度: {N} 字")
print(f"原文前50字: {text[:50]}...")

# 公式B：字符 embedding 表
unique, idx, all_embs = build_char_embeddings(text, window=4, dim=8)
n_unique = len(unique)
print(f"唯一字符数: {n_unique}")

# 把文章变成 embedding 序列
seq_embs = np.array([all_embs[idx[c]] for c in text])   # (N, 8)
x_axis   = np.arange(N, dtype=float)

# 测试不同 K 值
print(f"\n{'K':>5}  {'公式A大小':>9}  {'压缩率':>8}  {'字符准确率':>10}  还原前40字")
print("─" * 80)

for K in [20, 50, 100, 150, 200]:
    # 公式A：傅里叶拟合
    coeffs, T = fourier_fit(x_axis, seq_embs, K)

    # 还原
    recon_text, accuracy = reconstruct(
        text, seq_embs, coeffs, T, K, unique, all_embs)

    formula_A = (2*K+1) * all_embs.shape[1]   # 公式A大小
    compress  = (1 - formula_A / N) * 100
    preview   = recon_text[:40]

    print(f"{K:>5}  {formula_A:>9}  {compress:>7.1f}%  {accuracy:>9.1%}  {preview}")

# ─────────────────────────────────────────────
# 完整还原展示（K=150）
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  K=150 完整还原结果")
print("="*60)

K = 150
coeffs, T = fourier_fit(x_axis, seq_embs, K)
recon_text, accuracy = reconstruct(
    text, seq_embs, coeffs, T, K, unique, all_embs)

# 对比显示，标出错误位置
print(f"\n字符准确率: {accuracy:.1%}\n")
diff_display = []
errors = 0
for orig, recon in zip(text, recon_text):
    if orig == recon:
        diff_display.append(orig)
    else:
        diff_display.append(f"[{recon}]")
        errors += 1
diff_str = ''.join(diff_display)
print(f"原文 vs 还原（错误用[]标出）:\n{diff_str[:200]}...")
print(f"\n总错误字数: {errors}/{N}")

# ─────────────────────────────────────────────
# 存储分析
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  存储分析")
print("="*60)

dim         = all_embs.shape[1]
formula_A   = (2*150+1) * dim       # 公式A（傅里叶系数）
formula_B   = n_unique * dim        # 公式B（字符embedding表）
t_vals_size = N                     # t值（每字一个）
original    = N                     # 原文（每字一个整数）

print(f"""
  原始存储（直接存字符索引）  : {original:>5} 个数字
  ─────────────────────────────────────
  公式A（傅里叶系数，K=150）  : {formula_A:>5} 个数字  ← 跟文章长度有关
  公式B（字符embedding表）    : {formula_B:>5} 个数字  ← 跟文章长度有关

  公式B 的关键性质：
  → 只要语言相同，公式B 可以跨文章共用
  → 训练一次，所有文章共享同一张表
  → 每篇新文章只需存公式A

  如果公式B共用（不计入单篇成本）：
  单篇成本 = 公式A = {formula_A} 个数字
  原文大小 =         {original} 个数字
  压缩率   =         {(1-formula_A/original)*100:.1f}%（K=150时比原文大）

  甜蜜点（K=50）：
  公式A = {(2*50+1)*dim} 个数字，压缩率 = {(1-(2*50+1)*dim/original)*100:.1f}%
  代价  = 字符准确率下降（有损压缩）
""")
