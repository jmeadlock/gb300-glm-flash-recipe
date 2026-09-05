import numpy as np, sys
f = sys.argv[1]
a = np.fromfile(f, dtype=np.int16); L, K = 78, 8; n = a.size // (L * K); a = a[: n * L * K].reshape(n, L, K)
print("tokens", n, "min", a.min(), "max", a.max())
moe = [l for l in range(L) if (a[:, l, :] >= 0).any()]
print("moe layers", len(moe), "first", moe[:3], "last", moe[-3:])
rows = []
for l in moe:
    y = a[:, l, :]; y = y[y >= 0]
    c = np.bincount(y, minlength=256).astype(float); c /= c.sum(); s = np.sort(c)[::-1]; cum = np.cumsum(s)
    rows.append((l, cum[31], cum[63], cum[127], cum[159], int((c == 0).sum())))
print("layer  top32  top64  top128 top160 unused")
for r in rows[:6] + [rows[len(rows)//2]] + rows[-3:]:
    print(f"{r[0]:3d}   {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}  {r[5]}")
arr = np.array([r[1:5] for r in rows])
print("MEAN all-moe  top32=%.3f top64=%.3f top128=%.3f top160=%.3f" % tuple(arr.mean(0)))
print("MEAN first42  top32=%.3f top64=%.3f top128=%.3f top160=%.3f" % tuple(arr[:42].mean(0)))
print("uniform would be: top32=0.125 top64=0.250 top128=0.500 top160=0.625")
