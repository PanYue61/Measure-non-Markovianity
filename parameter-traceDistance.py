import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import streamlit as st

if "curves" not in st.session_state:
    st.session_state.curves = []

def compute_traceDistance(p0,p1,p2,p3,q):
    ptotal = p0 + p1 + p2 + p3
    p0 = p0 / ptotal
    p1 = p1 / ptotal
    p2 = p2 / ptotal
    p3 = p3 / ptotal
    ii = (p1 + p2) * (p0 + p3) * q
    jj = -(p1 + p2) * (p0 + p3) * q
    kk = (p0 - p3) * (p1 + p2) * q
    ll = (p2 - p1) * (p0 + p3) * q
    mm = (p3 - p0) * (p1 + p2) * q
    nn = (p2 - p1) * (p0 - p3) * q
    oo = (p0 + p3) * (p1 + p2) * q
    pp = (p1 - p2) * (p0 + p3) * q
    qq = (p0 + p3 - (p0 - p3) ** 2) * q
    rr = (p1 + p2 - (p1 - p2) ** 2) * q
    W = np.array([
        [ii, 0, 0, kk, 0, 0, 0, 0, 0, 0, 0, 0, kk, 0, 0, qq],
        [0, jj, ll, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mm, nn, 0],
        [0, ll, jj, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nn, mm, 0],
        [kk, 0, 0, ll, 0, 0, 0, 0, 0, 0, 0, 0, qq, 0, 0, kk],
        [0, 0, 0, 0, jj, 0, 0, mm, ll, 0, 0, nn, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, oo, pp, 0, 0, pp, rr, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, pp, oo, 0, 0, rr, pp, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, mm, 0, 0, jj, nn, 0, 0, ll, 0, 0, 0, 0],
        [0, 0, 0, 0, ll, 0, 0, nn, jj, 0, 0, mm, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, pp, rr, 0, 0, oo, pp, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, rr, pp, 0, 0, pp, oo, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, nn, 0, 0, ll, mm, 0, 0, jj, 0, 0, 0, 0],
        [kk, 0, 0, qq, 0, 0, 0, 0, 0, 0, 0, 0, ii, 0, 0, kk],
        [0, mm, nn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, jj, ll, 0],
        [0, nn, mm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ll, jj, 0],
        [qq, 0, 0, kk, 0, 0, 0, 0, 0, 0, 0, 0, kk, 0, 0, ii]
    ], dtype=float)
    M = W @ W.T
    sqrtM = sqrtm(M)
    return 0.5*np.trace(sqrtM)


st.title("BLP measurement vs p0")
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        q = st.number_input("q", 0.0, 1.0, 0.5, step=0.01)
    with c2:
        p1 = st.number_input("p1", 0.0, 200.0, 30.0, step=0.1)
    with c3:
        p2 = st.number_input("p2", 0.0, 200.0, 45.0, step=0.1)
    with c4:
        p3 = st.number_input("p3", 0.0, 200.0, 35.0, step=0.1)

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🔄 清空历史曲线"):
        st.session_state.curves = []
with col_b:
    keep = st.checkbox("✅ 保留当前曲线", value=False)

c5, c6, c7, c8 = st.columns(4)
with c5:
    x_min = st.number_input("x_min", value=0.0)
with c6:
    x_max = st.number_input("x_max", value=200.0)
with c7:
    y_min = st.number_input("y_min", value=0.0)
with c8:
    y_max = st.number_input("y_max", value=2.0)

p0s = np.linspace(1, 200, 500)
ys = [compute_traceDistance(p0, p1, p2, p3, q) for p0 in p0s]
if keep:
    st.session_state.curves.append((p0s, ys, f"q={q}, p1={p1}, p2={p2}, p3={p3}"))

# 绘图

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.plot(p0s, ys, label="Current Curve")
for curve in st.session_state.curves:
    ax.plot(curve[0], curve[1], linestyle='--', label=curve[2])

ax.set_xlim(1, 200)
ax.set_ylim(0, 2)
ax.set_xlabel("p0")
ax.set_ylabel("BLP Measurement")
ax.set_title("Trace Distance vs p0")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
fig.subplots_adjust(right=0.75)
ax.grid(True)
st.pyplot(fig)

# x = np.linspace(0, 10, 100)
# y = np.sin(x * q)
#
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title(f"q = {q}")
# st.pyplot(fig)

# results1 = []
# results2 = []
# results3 = []
# results4 = []
# results5 = []
# results6 = []

# qs = np.linspace(0.5, 0.5, 1)
# p0s = np.linspace(1, 300, 500)
# p1s = np.linspace(30, 30, 1)
# p2s = np.linspace(45, 45, 1)
# p3s = np.linspace(35, 35, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results1.append((q, p0, p1, p2, p3, val))
#                     # print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
# qs = np.linspace(0.5, 0.5, 1)
# p0s = np.linspace(1, 300, 500)
# p1s = np.linspace(5, 5, 1)
# p2s = np.linspace(45, 45, 1)
# p3s = np.linspace(150, 150, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results2.append((q, p0, p1, p2, p3, val))
#                     print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
# qs = np.linspace(0.4, 0.4, 1)
# p0s = np.linspace(1, 300, 500)
# p1s = np.linspace(150, 150, 1)
# p2s = np.linspace(150, 150, 1)
# p3s = np.linspace(150, 150, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results3.append((q, p0, p1, p2, p3, val))
#                     print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
# qs = np.linspace(0.6, 0.6, 1)
# p0s = np.linspace(1, 100, 100)
# p1s = np.linspace(30, 30, 1)
# p2s = np.linspace(45, 45, 1)
# p3s = np.linspace(35, 35, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results4.append((q, p0, p1, p2, p3, val))
#                     print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
# qs = np.linspace(0.8, 0.8, 1)
# p0s = np.linspace(1, 100, 100)
# p1s = np.linspace(30, 30, 1)
# p2s = np.linspace(45, 45, 1)
# p3s = np.linspace(35, 35, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results5.append((q, p0, p1, p2, p3, val))
#                     print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
# qs = np.linspace(1, 1, 1)
# p0s = np.linspace(1, 100, 100)
# p1s = np.linspace(30, 30, 1)
# p2s = np.linspace(45, 45, 1)
# p3s = np.linspace(35, 35, 1)
#
# for p0 in p0s:
#     for p1 in p1s:
#         for p2 in p2s:
#             for p3 in p3s:
#                 for q in qs:
#                     val = compute_traceDistance(p0, p1, p2, p3, q)
#                     results6.append((q, p0, p1, p2, p3, val))
#                     print(f"q = {q:.2f}, p0 = {p0}, p1 = {p1}, p2 = {p2}, p3 = {p3}, result = {val:.4f}")
#
#
# x1 = [row[1] for row in results1]
# y1 = [row[5] for row in results1]
#
# x2 = [row[1] for row in results2]
# y2 = [row[5] for row in results2]
#
# x3 = [row[1] for row in results3]
# y3 = [row[5] for row in results3]
#
# x4 = [row[1] for row in results4]
# y4 = [row[5] for row in results4]
#
# x5 = [row[1] for row in results5]
# y5 = [row[5] for row in results5]
#
# x6 = [row[1] for row in results6]
# y6 = [row[5] for row in results6]
#
# plt.plot(x1, y1, label='Group 1')
# plt.plot(x2, y2, label='Group 2')
# plt.plot(x3, y3, label='Group 3')
# # plt.plot(x4, y4, label='Group 1')
# # plt.plot(x5, y5, label='Group 2')
# # plt.plot(x6, y6, label='Group 3')
#
# plt.xlabel('p0')  # 如果你是在扫 p0
# plt.ylabel('2 * Tr(sqrt(W Wᵗ))')
# plt.title('Trace Distance vs p0 for different groups')
# plt.grid(True)
# plt.legend()  # 显示图例
# plt.show()