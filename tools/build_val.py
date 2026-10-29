"""Builds validation.ipynb: all correctness checks (D4 code / protocol, 15-to-1 / GoSC layout, noisy simulation)."""
from nbutil import *

cells = [md(r'''
# validation — 正しさの検証（計算資源の比較とは独立）

`D4_noisy_simulation.ipynb`・`noisy_cost_comparison.ipynb` が使う構成要素が正しいことを、比較の計算とは**独立に**確認する。
各検査は `record(名前, 合否, 詳細)` で記録し（数値を記録するだけで合否の無い項目は `INFO`）、最後に `results/validation.json` 1 個にまとめて保存する。
第 1 部・第 2 部は v4 から引き継いだ雑音なしの検証、第 3 部は v5 で追加した雑音付きシミュレーションの検証。所要時間は CPU 1 コアで約 30 分。

| ID | 内容 | 方法 |
|---|---|---|
| **第 1 部** | **D4 符号・code switching（[KZH]）** | |
| V1 | 安定化子の定義が [KZH] と一致 | Fig. 4 から読み取った特徴との照合 |
| V2 | 符号空間の整合性 | $C_Z$ 保存・可換性・$S^2=1$・$k=1$（$\mathbb F_2$ 厳密）＋負の対照 |
| V3 | 独立実装による照合 | $L=1$ を密ベクトルで独立に実装 |
| V4 | 論理演算子と距離 | MILP の距離を全探索で照合 |
| V5, V5b | Knill–Laflamme 条件 | $L=2$ の全 weight≤2 Pauli；D4 相の距離 $=L$（$L\le5$） |
| V6 | transversal $U=WV$ | 局所厳密列挙、論理位相、負の対照 |
| V7, V7b, V8 | プロトコル全体 | $L=1,2$ 厳密シミュレーション；$L=3,4$ 回路と位相公式の照合；$L\le8$ 局所厳密証明 |
| V9, V10, V11 | 測定ガジェット・スケジュール・計数 | 密行列、可換性、幾何からの直接公式 |
| **第 2 部** | **15-to-1・GoSC の配置（[Lit]）** | |
| M1–M5 | 15-to-1 の行列・回路・誤り検出、Fig. 17a/b の回転 | 状態ベクトル |
| M6–M9 | tile 数・時間、物理レベルの格子手術（Fig. 17c の 11 個の多パッチ測定） | $\mathbb F_2$、MILP、stim |
| **第 3 部** | **雑音付きシミュレーション（v5）** | |
| N1 | Takada らの方法の十分条件（Definition 2, Sec. VIII）の、D4 code switching の生成元・論理演算子についての代数的な確認 | $\mathbb F_2$ の厳密判定（$L=2..7$ と 9。有限サイズの確認であり一般の $L$ の証明ではない）、演算子の比較は大域位相だけを許す厳密比較＋負の対照 |
| N2 | Pauli proxy の「ドレッシング」$C_e=D^\dagger X^eDX^e$ が Clifford で、実装したゲート列と一致 | 全ての非 Clifford ゲート・全パターンの行列 |
| N3 | **Pauli proxy 回路（Clifford）と真の非 Clifford 回路の一致** | 同じ誤り・同じ補正で並走させ、全測定の確率と最終 $\langle\bar H\rangle$ を比較：(a) $L=1,2$、ランダムな $X$ 補正；(b) $L=2$、**本番と同じ適応処理**（flux 復号の補正、switch-out の復号済み読み出しと物理 CZ 層とその雑音、Pauli frame） |
| N4 | 雑音なしで出力誤り 0（復号器込み） | $d=3,5,7$ |
| N5 | 1 個の故障に対する結果（INFO：故障の網羅範囲と失敗の記録） | switch-out の動的な CZ・アイドルを含む全雑音位置×全 Pauli、各 2 つの測定乱数の種；$d=3$ 全列挙（A, B）、$d=5$ 抽出。確率で重み付けした一次の係数 $c_1$（$\varepsilon\approx c_1p$）を窓の内外に分けて記録 |
| N9 | switch-out の物理スケジュール | 補正 CZ の層が正しい（同じ量子ビットを同時に使わない）、適用される CZ が $\prod_{v\in\kappa}S^g_v$ の CZ 部分（偶数回は相殺）と一致、時間の計数 |
| N6 | Litinski（2019）の誤り予算の再実装が公開値を再現 | 公開ノートブックに保存された出力 8 件 |
| N7 | 較正用の表面符号回路 | 雑音なしで検出器 0、回路レベルの距離 $=d$、1 ラウンド 6 時間ステップ |
| N8 | GoSC 15-to-1 の誤り予算モデル | 雑音なしで誤り 0、入力 $Z$ 誤りだけなら $35p^3$ |
'''),
    code(SETUP), code(INSTALL),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'msd_lib.py', 'noisy_d4_lib.py', 'msd_noisy_lib.py'),
    code(r'''import time, itertools
VAL = {}
def record(name, passed, detail=''):
    detail = str(detail).replace('\n', ' ')
    VAL[name] = dict(status='PASS' if passed else 'FAIL', detail=detail)
    print(('PASS ' if passed else 'FAIL ') + name + (f'  | {detail}' if detail else ''))
def info(name, detail):
    VAL[name] = dict(status='INFO', detail=str(detail))
    print('INFO ' + name + '  | ' + str(detail))
QUICK = False
L_MAX_INVARIANCE = 5 if QUICK else 8
rng = np.random.default_rng(SEED)'''),
    md(r'''# 第 1 部 D4 符号と code switching（雑音なし、v4 から引き継ぎ）'''),
    md(r'''## V1. 安定化子の定義と [KZH] Fig. 4 の照合

Fig. 4（$L=9$）から読み取れる事項を、コードから得た構造と照合する。
* 左境界 $\mathcal L_r$ 上：緑・青の $X$ 安定化子（3辺）。赤は無し。
* 上境界 $\mathcal L_b$ 上：赤・緑。左上角：緑（2辺）。
* 下・右境界 $\mathcal L_{rb}$：赤青の積 $XX$。右下角も $XX$（2辺）。
* 境界 $Z$：$\mathcal L_r$ に $Z^r$、$\mathcal L_b$ に $Z^b$、$\mathcal L_{rb}$ に $Z^rZ^b$ と $Z^g$。
* バルクの CZ：$S^r$ は右上 plaquette に CZ$^{g,b}$ 2本、$S^g$ は左上・右下 plaquette に CZ$^{r,b}$ 各1本、$S^b$ は左下 plaquette に CZ$^{r,g}$ 2本。'''),
    code(r'''lat = D4Lattice(9); L = 9
exp = {'bulk': [('r',), ('g',), ('b',)], 'Lr': [('g',), ('b',)], 'Lb': [('r',), ('g',)], 'TL': [('g',)],
       'Lrb': [('r', 'b')], 'BL': [], 'TR': []}
def vclass(v):
    x, y = v
    if v == (0, 0): return 'BL'
    if v == (0, L): return 'TL'
    if v == (L, L): return 'TR'
    if 0 < x < L and 0 < y < L: return 'bulk'
    if x == 0: return 'Lr'
    if y == L: return 'Lb'
    return 'Lrb'
ok = True
for v in lat.vertices:
    got = [tuple(CNAME[c] for c in combo) for combo in lat.vertex_x_types(v)]
    if sorted(got) != sorted(exp[vclass(v)]):
        ok = False; print('mismatch at', v, got)
record('V1a boundary X-generator types match Fig. 4', ok)
zl = lat.z_checks()
bd = Counter((lab[2], lat.boundary_of_edge(lab[1])) for lab, _ in zl if lab[0] == 'bdZ')
exp_bd = {('r', 'Lr'): L, ('b', 'Lb'): L, ('rb', 'Lrb'): 2 * L, ('g', 'Lrb'): 2 * L}
record('V1b boundary Z checks (Z^r on L_r, Z^b on L_b, Z^rZ^b and Z^g on L_rb)', dict(bd) == exp_bd, dict(bd))
def rel_plaq(v, c):
    """plaquette offsets (relative to v) that carry the CZ dressing of S^c_v"""
    out = []
    for (i, j) in lat.SX(v, c).quad:
        es = {i // 3, j // 3}
        for p in lat.plaquettes:
            if es <= set(lat.plaq_edges(p)):
                out.append((p[0] - v[0], p[1] - v[1]))
    return Counter(out)
v0 = (4, 4)
ok = (rel_plaq(v0, R) == Counter({(0, 0): 2}) and rel_plaq(v0, G) == Counter({(-1, 0): 1, (0, -1): 1})
      and rel_plaq(v0, B) == Counter({(-1, -1): 2}))
record('V1c bulk CZ positions (S^r: upper-right x2, S^g: upper-left + lower-right, S^b: lower-left x2)', ok,
       {c: dict(rel_plaq(v0, c)) for c in (R, G, B)})
colpairs = {c: {tuple(sorted((i % 3, j % 3))) for (i, j) in lat.SX(v0, c).quad} for c in (R, G, B)}
record('V1d CZ colour pairs (S^r: g-b, S^g: r-b, S^b: r-g)', colpairs == {R: {(G, B)}, G: {(R, B)}, B: {(R, G)}})'''),
    md(r'''## V2. 符号空間の整合性（$\mathbb F_2$ 上の厳密判定）

$X$ 型 Clifford 安定化子 $A,B$ について $AB=(-1)^{g(x)}BA$ の $g$ はアフィン関数になる（2次項は相殺）。$C_Z$ 上で $g\equiv 0$ ⇔ 定数項0かつ線形部分が $H_Z$ の行空間に属する。
**負の対照**：FSC（$g$ を除いた2層）に CZ 修飾付きの生成元を使うと、$g$ 量子ビットの $Z$ 安定化子が無いため非可換になるはず → 検出できることを確認。'''),
    code(r'''for L in range(1, 7):
    cd = CodeData(D4Lattice(L))
    r = code_consistency(cd)
    record(f'V2 D4 L={L}: shifts in C_Z, commuting on C_Z, S^2=1, k=1',
           r['shift_in_CZ'] and not r['noncommuting_pairs'] and not r['bad_squares'] and r['k'] == 1 and r['shifts_independent'],
           {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()})
    rf = code_consistency(CodeData(D4Lattice(L), (R, B), dressed=False))
    record(f'V2 FSC L={L}: CSS code, k=1', rf['shift_in_CZ'] and not rf['noncommuting_pairs'] and rf['k'] == 1)
neg = code_consistency(CodeData(D4Lattice(3), (R, B), dressed=True))
record('V2 negative control: dressed generators without g-qubit Z checks are detected as non-commuting',
       len(neg['noncommuting_pairs']) > 0, f"{len(neg['noncommuting_pairs'])} non-commuting pairs")'''),
    md(r'''## V3. 独立実装（$L=1$、12量子ビット密ベクトル）

埋め込みの D4 ライブラリの関数を一切使わず、式(2)(3)(7)(8) を直接 4096 次元のベクトル演算として実装する。$Z$ 部分空間の計算基底を列挙し、その上で $S|\psi\rangle=|\psi\rangle$ の解空間を SVD で求め、論理 $Z$ 演算子を重み≤2 の $Z$ 型 Pauli から探索し、$U=WV$ の論理行列を求める。'''),
    code(r'''n1 = 12; N1 = 2 ** n1; idx1 = np.arange(N1)
qq = lambda e, c: 3 * e + c        # edges: 0 bottom h(0,0), 1 top h(0,1), 2 left v(0,0), 3 right v(1,0)
bit1 = lambda i, k: (i >> (n1 - 1 - k)) & 1
def Zop(qs):
    par = np.zeros(N1, dtype=int)
    for k in qs: par ^= bit1(idx1, k)
    return lambda v: (1 - 2 * par) * v
def Xop(qs):
    m = 0
    for k in qs: m |= 1 << (n1 - 1 - k)
    return lambda v: v[idx1 ^ m]
def CZop(pairs):
    par = np.zeros(N1, dtype=int)
    for a, b in pairs: par ^= bit1(idx1, a) & bit1(idx1, b)
    return lambda v: (1 - 2 * par) * v
def compose(*fs):
    def h(v):
        for f in reversed(fs): v = f(v)
        return v
    return h
Zchecks = [[qq(e, c) for e in range(4)] for c in range(3)]
Zchecks += [[qq(2, 0)], [qq(1, 2)], [qq(0, 0), qq(0, 2)], [qq(3, 0), qq(3, 2)], [qq(0, 1)], [qq(3, 1)]]
S_rb = compose(Xop([qq(0, 0), qq(3, 0)]), Xop([qq(0, 2), qq(3, 2)]))           # vertex (1,0) on L_rb
S_g = compose(Xop([qq(2, 1), qq(1, 1)]), CZop([(qq(2, 0), qq(1, 2))]))          # vertex (0,1), v = vertex 2 of p
okZ = np.ones(N1, bool)
for zc in Zchecks:
    par = np.zeros(N1, dtype=int)
    for k in zc: par ^= bit1(idx1, k)
    okZ &= par == 0
basis = np.nonzero(okZ)[0]
Bm = np.zeros((N1, len(basis)), complex); Bm[basis, np.arange(len(basis))] = 1
rows_ = []
for S in (S_rb, S_g):
    M = np.stack([S(Bm[:, j]) for j in range(Bm.shape[1])], 1)
    assert np.allclose(M[~okZ], 0)
    rows_.append(Bm.conj().T @ M - np.eye(len(basis)))
u_, s_, vh_ = np.linalg.svd(np.concatenate(rows_))
code1 = Bm @ vh_[np.sum(s_ > 1e-9):].conj().T
record('V3a independent L=1: code dimension = 2', code1.shape[1] == 2, f'dim C_Z={len(basis)}')
P1 = code1 @ code1.conj().T
found = None
for w in (1, 2):
    for qs in itertools.combinations(range(n1), w):
        M = np.stack([Zop(qs)(code1[:, j]) for j in range(2)], 1)
        if np.allclose(P1 @ M, M):
            Mc = code1.conj().T @ M
            if not np.allclose(Mc, Mc[0, 0] * np.eye(2)):
                found = (qs, Mc); break
    if found: break
ev, evec = np.linalg.eigh(found[1]); logical1 = code1 @ evec[:, ::-1]
def CNOT1(c, t):
    return lambda v: np.where(bit1(idx1, c) == 1, v[idx1 ^ (1 << (n1 - 1 - t))], v)
V1 = compose(*[CNOT1(qq(e, c), qq(e, 1)) for e in range(4) for c in (0, 2)])
def phase1(terms):
    ph = np.zeros(N1)
    for c, qs in terms:
        m = np.ones(N1, dtype=int)
        for k in qs: m &= bit1(idx1, k)
        ph = ph + c * m
    return lambda v: np.exp(1j * np.pi / 4 * ph) * v
W1 = phase1([(2, (qq(0, 0), qq(3, 2))), (-2, (qq(2, 0), qq(1, 2)))] + [(-1, (qq(e, 0),)) for e in (0, 3)])
U1 = compose(W1, V1)
UL = np.stack([U1(logical1[:, j]) for j in range(2)], 1)
Um = logical1.conj().T @ UL; Um = Um / Um[0, 0]
record('V3b independent L=1: U preserves the code space', np.allclose(P1 @ UL, UL))
record('V3c independent L=1: logical action of U = T^dag (Z-basis defined by the trivial gauge field)',
       np.allclose(Um, np.diag([1, np.exp(-1j * np.pi / 4)])), np.round(Um, 4))'''),
    md(r'''## V4. 論理演算子と距離（MILP を全探索で照合）
$d_Z$：$Su=0$, $t_{\log}\cdot u=1$ を満たす最小重み $u$。$d_X^{\rm shift}$：$H_Z t=0$, $z_{\log}\cdot t=1$ を満たす最小重み $t$。frozen 量子ビットを含む最小解は無い（frozen 上の $Z$ は安定化子、$X$ は検出される）ので frozen を除いて全探索する。'''),
    code(r'''def brute_min(Hrows, lvec, cols, wmax):
    Hc = Hrows[:, cols].astype(np.int64); lc = lvec[cols].astype(np.int64)
    for w in range(1, wmax + 1):
        for sup in itertools.combinations(range(len(cols)), w):
            sup = list(sup)
            if (lc[sup].sum() % 2 == 1) and not (Hc[:, sup].sum(axis=1) % 2).any():
                return w
    return None
for L, kind in [(2, 'Z'), (3, 'Z'), (2, 'X')]:
    lat = D4Lattice(L); fz = set()
    for lab, rows in lat.z_checks():
        if lab[0] == 'bdZ' and len(rows) == 1: fz |= rows
    for name, cd in [('FSC', CodeData(lat, (R, B), dressed=False)), ('D4', CodeData(lat))]:
        cols = [c for c in cd.cols if c not in fz]
        dz, dx, _, _ = distances(cd)
        if kind == 'Z':
            bf = brute_min(cd.S, cd.t_log, cols, dz)
            record(f'V4 {name} L={L}: brute-force d_Z = MILP d_Z = L', bf == dz == L, (bf, dz))
        else:
            if len(cols) > 50:
                continue
            bf = brute_min(cd.HZ, cd.z_log, cols, dx)
            record(f'V4 {name} L={L}: brute-force X-shift distance = MILP = L+3', bf == dx == L + 3, (bf, dx))'''),
    md(r'''## V5. Knill–Laflamme 条件（$L=2$）
$\langle i|E|j\rangle = c_E\delta_{ij}$ を、論理基底 $|0_L\rangle,|1_L\rangle$（各 $2^{11}$ 項の疎ベクトル）と全ての weight≤2 Pauli $E$ について直接検査する（約1分）。D4 相の真の距離が $L=2$ で 2 であることを確認する（$L\ge3$ は**未検証**）。'''),
    code(r'''t0 = time.time()
lat2 = D4Lattice(2); d42 = CodeData(lat2)
kb = [coset_state(d42, set()), coset_state(d42, set(np.nonzero(d42.t_log)[0]))]
def apply_pauli(st, xs, zs):
    xs = frozenset(xs); out = {}
    for x, v in st.a.items():
        out[x ^ xs] = v * (-1) ** len(x & zs)
    return SparseState(out)
viol = {}
for w in (1, 2):
    viol[w] = 0
    for qs in itertools.combinations(range(lat2.n), w):
        for ps in itertools.product('XYZ', repeat=w):
            xs = [a for a, p in zip(qs, ps) if p in 'XY']; zs = {a for a, p in zip(qs, ps) if p in 'YZ'}
            M = np.array([[kb[i].overlap(apply_pauli(kb[j], xs, zs)) for j in range(2)] for i in range(2)])
            if abs(M[0, 1]) > 1e-9 or abs(M[1, 0]) > 1e-9 or abs(M[0, 0] - M[1, 1]) > 1e-9:
                viol[w] += 1
record('V5 D4 L=2 Knill-Laflamme: all weight-1 Paulis correctable-detectable, some weight-2 logical (distance = 2)',
       viol[1] == 0 and viol[2] > 0, f'violations {viol}, {time.time() - t0:.0f}s')'''),
    md(r"""## V5b. D4 相の Knill–Laflamme 距離 $=L$ の厳密証明（$L\le 5$、`FULL_KL=True` で $L=6$）

$E=X^tZ^u$ について：(i) $t\notin C_Z$ なら $E$ は符号空間を直交空間へ写す（検出可能）。(ii) $t$ が論理類なら $|t|\ge L+3$（V4）。(iii) $t\in\mathrm{span}(S)$ なら $X^t=S_cF_c$（$S_c$ は安定化子の積、$F_c=(-1)^{f_c}$ は対角）なので $PEP=PF_cZ^uP$。$F_c$ を $Z$ 型 Pauli $Z^v$（$v\subseteq Q_f=\mathrm{supp}f_c$）に展開すると、KL 条件が破れるには $v+u$ が論理 $Z$ でなければならない。よって
$$|E|\ \ge\ |t| + \min_{\ell:\ \text{logical }Z}|\ell\setminus(Q_f\cup t)|=:LB_t .$$
重み $\le L-1$ の全ての $t\in\mathrm{span}(S)$ を MILP（no-good cut 付き）で**厳密に列挙**し、$\min_t LB_t\ge L$ を確認する。$t=0$ は $d_Z=L$ で扱える。重み $L$ の論理 $Z$ があるので距離はちょうど $L$。"""),
    code(r"""FULL_KL = False
for L in range(3, 7 if FULL_KL else 6):
    t0 = time.time()
    P_ = D4Protocol(L)
    lb, ts, det, stop = kl_distance_lower_bound(P_.d4, L - 1, P_.frozen)
    dz, dx, _, _ = distances(P_.d4)
    # stop is 'infeasible' or 'min weight > wmax': the enumeration ended because the solver
    # proved there is no further t (any other solver status raises inside _milp_binary)
    ok = stop in ('infeasible', 'min weight > wmax') and (lb is None or lb >= L) and dz == L and dx >= L
    record(f'V5b L={L}: every Pauli error of weight <= L-1 satisfies Knill-Laflamme -> D4-phase distance = L',
           ok, f'{len(ts)} stabilizer shifts of weight <= {L-1} enumerated (stop: {stop}); min LB_t = {lb}; d_Z = {dz}; X-shift = {dx}; {time.time()-t0:.0f}s')"""),
    md(r'''## V6. transversal $U=WV$（式(5)(8)）

各生成元 $S$ について、[KZH] SM の式（permuting stabilizers）が予言する像 $S'$（$S^r\to S^rS^g$, $S^g\to S^g$, $S^b\to S^gS^b$, $\mathcal L_{rb}$ 上の $S^rS^b\to S^rS^b$）と $USU^\dagger$ が $C_Z$ 上で一致することを、**位相差が依存する量子ビットへの $C_Z$ の射影を厳密に全列挙**して確認する。
負の対照：(i) 境界の $T^\dagger$ を除く、(ii) CS と CS$^\dagger$ を入れ替える、(iii) 境界を $T$ にする、(iv) 両方 CS にする → 失敗するか論理作用が変わるはず。'''),
    code(r'''def check_U(L, cs=(1, -1), bt=-1):
    lat = D4Lattice(L); cd = CodeData(lat); Wop = W_operator(lat, cs, bt)
    fails, mx = [], 0
    for lab, op in cd.xgens:
        ok, n = check_conjugation(cd, Wop, op, U_target(lat, lab))
        mx = max(mx, n)
        if not ok: fails.append(lab)
    return fails, mx, logical_phases(cd, Wop), cd
for L in range(1, 7):
    fails, mx, ph, cd = check_U(L)
    # V maps C_Z into C_Z
    Bz = CZ_basis(cd)
    VB = np.array([cd.lat.vec(V_map(cd.lat, frozenset(np.nonzero(b)[0]))) for b in Bz])
    vz = not ((cd.HZ.astype(int) @ VB.T.astype(int)) % 2).any()
    record(f'V6 L={L}: U S U^dag = S\' on C_Z for all {len(cd.xgens)} generators; V(C_Z)=C_Z; logical phases (pi/4 units) = [0,7] (T^dag)',
           not fails and vz and ph == [0, 7], f'max local configs {mx}, phases {ph}')
for name, cs, bt in [('no boundary T^dag', (1, -1), 0), ('CS<->CS^dag swapped', (-1, 1), -1),
                     ('boundary T instead of T^dag', (1, -1), 1), ('CS,CS (no dagger)', (1, 1), -1)]:
    fails, mx, ph, _ = check_U(4, cs, bt)
    record(f'V6 negative control L=4 [{name}] is rejected or changes the logical action', bool(fails) or ph != [0, 7],
           f'{len(fails)} failing generators, phases {ph}')
fails, mx, ph, _ = check_U(4, (-1, 1), 1)
record('V6 conjugate convention (CS^dag,CS, boundary T) gives a valid logical T (consistency)', not fails and ph == [0, 1], ph)'''),
    md(r'''## V7. プロトコル全体の厳密シミュレーション（$L=1,2$）

[KZH] の手順を疎状態ベクトル（計算基底の辞書）でそのまま実行する。
1. FSC の論理状態 $\alpha|0_L\rangle+\beta|1_L\rangle$（ランダム）、$g$ は $|0\rangle$。全 $Z$ 安定化子が $+1$ であることを確認。
2. D4 の全 $X$ 型生成元を射影測定（ボルン則で分岐をサンプル）。2回目の測定で結果が再現すること（ノイズなしで1ラウンドで十分）を確認。
3. $U=WV$。
4. 全 $Z^g_e$ を測定し、$\delta\kappa=a_g$ を解いて $\prod_{v\in\kappa}S^g_{X;v}$ を適用。
5. $m_v=1$ の各頂点に $\mathcal L_{rb}$ から $Z^rZ^b$ 文字列（Pauli フレーム補正）。
6. 目標状態 $\alpha|0_L\rangle+e^{-i\pi/4}\beta|1_L\rangle$ との忠実度。

負の対照：補正5を外す／補正4の CZ 部分を外す（$X^g$ のみ）／境界 $T^\dagger$ を外す。'''),
    code(r'''def run_protocol(L, alpha, beta, rng, skip_pauli=False, sg_x_only=False, W_bt=-1):
    lat = D4Lattice(L); fsc = CodeData(lat, (R, B), dressed=False); d4 = CodeData(lat)
    zero = coset_state(fsc, set()); one = coset_state(fsc, set(np.nonzero(fsc.t_log)[0]))
    psi = SparseState({k: alpha * v for k, v in zero.a.items()})
    for k, v in one.a.items(): psi.a[k] = psi.a.get(k, 0) + beta * v
    for lab, rows in lat.z_checks():
        assert abs(psi.expect(PSOp(lin=rows)) - 1) < 1e-9
    m = {}
    for lab, op in d4.xgens:
        bit, psi, p = psi.measure_op(op, rng); m[lab] = bit
    repeat_ok = all(abs(psi.expect(op) - (-1) ** m[lab]) < 1e-9 for lab, op in d4.xgens)
    zero_ok = all(abs(psi.expect(PSOp(lin=rows)) - 1) < 1e-9 for lab, rows in lat.z_checks())
    psi = psi.apply_perm(lambda x: V_map(lat, x)).apply_diag(W_operator(lat, (1, -1), W_bt))
    ag = set()
    for e in range(lat.E):
        bit, psi, p = psi.measure_z(lat.q(e, G), rng)
        if bit: ag.add(e)
    ggens = [(lab, op) for lab, op in d4.xgens if lab[2] == 'g']
    Sg = np.array([lat.vec(op.s) for _, op in ggens])
    kappa = f2_solve(Sg, lat.vec({lat.q(e, G) for e in ag}))
    for i in np.nonzero(kappa)[0]:
        op = ggens[i][1]
        psi = psi.apply_psop(PSOp(s=op.s) if sg_x_only else op)
    gclean = all(all(lat.q(e, G) not in x for e in range(lat.E)) for x in psi.a)
    if not skip_pauli:
        for lab, op in ggens:
            if m[lab]:
                x0, y0 = lab[1]
                qs = {lat.q(lat.eidx[('v', x0, y)], c) for y in range(y0) for c in (R, B)}
                psi = psi.apply_diag(DiagPhase([(4, (q,)) for q in qs]))
    target = SparseState({k: alpha * v for k, v in zero.a.items()})
    for k, v in one.a.items(): target.a[k] = target.a.get(k, 0) + beta * OMEGA ** 7 * v
    F = abs(target.overlap(psi)) ** 2 / (target.norm2() * psi.norm2())
    mg = tuple(m[lab] for lab, _ in ggens)
    return dict(F=F, repeat_ok=repeat_ok, zero_ok=zero_ok, g_cleared=gclean, m_g=mg,
                m_other_nontrivial=any(m[lab] for lab in m if lab[2] != 'g'), kappa_weight=int(kappa.sum()),
                n_g=len(ggens))
def rand_state(rng):
    a, b = rng.normal(size=2) + 1j * rng.normal(size=2); nrm = np.hypot(abs(a), abs(b)); return a / nrm, b / nrm
stats = {}
for L, ntr in [(1, 60), (2, 200 if not QUICK else 40)]:
    t0 = time.time(); res = [run_protocol(L, *rand_state(rng), rng) for _ in range(ntr)]
    Fmin = min(r['F'] for r in res)
    record(f'V7 L={L}: exact protocol, {ntr} random inputs/branches -> fidelity 1 with T^dag action',
           Fmin > 1 - 1e-9 and all(r['repeat_ok'] and r['zero_ok'] and r['g_cleared'] for r in res)
           and not any(r['m_other_nontrivial'] for r in res), f'min F = {Fmin:.12f}, {time.time() - t0:.0f}s')
    stats[L] = res
# basis inputs (logical information preserved for |0>, |1>, |+>, |+i>)
for name, (a, b) in {'|0>': (1, 0), '|1>': (0, 1), '|+>': (2 ** -.5, 2 ** -.5), '|+i>': (2 ** -.5, 1j * 2 ** -.5)}.items():
    Fs = [run_protocol(2, a, b, rng)['F'] for _ in range(10)]
    record(f'V7 L=2 input {name}: output = T^dag {name}', min(Fs) > 1 - 1e-9, f'min F={min(Fs):.12f}')
res2 = stats[2]; ng = res2[0]['n_g']
cm = Counter(r['m_g'] for r in res2); ck = Counter(r['kappa_weight'] for r in res2)
from scipy.stats import chisquare
p_m = chisquare([cm.get(t, 0) for t in itertools.product((0, 1), repeat=ng)]).pvalue
record(f'V7 L=2: S^g outcomes m are random (all {2**ng} patterns occur; chi^2 test vs uniform p={p_m:.2f})',
       len(cm) == 2 ** ng and p_m > 1e-3, dict(sorted(cm.items())))
mean_k = np.mean([r['kappa_weight'] for r in res2])
record('V7 L=2: switch-out correction |kappa| has mean ~ n_g/2 (supports expected CZ count = worst/2)',
       abs(mean_k - ng / 2) < 0.35, f'mean |kappa| = {mean_k:.2f}, n_g = {ng}, hist {dict(sorted(ck.items()))}')
negs = {'without Z^rZ^b Pauli-frame correction': dict(skip_pauli=True),
        'S^g correction without its CZ part': dict(sg_x_only=True),
        'W without boundary T^dag': dict(W_bt=0)}
for name, kw in negs.items():
    Fs = [run_protocol(2, *rand_state(rng), rng, **kw)['F'] for _ in range(60)]
    record(f'V7 negative control [{name}] fails for some branches', min(Fs) < 1 - 1e-6, f'min F = {min(Fs):.4f}')'''),
    md(r"""## V7b. プロトコル（回路）と位相公式 $F(a)$ の照合（$L=3,4$）

V8 の局所証明は配置ごとの位相の式 $F(a)$ に基づく。その式自体を、$L=1,2$ の符号状態シミュレーション（V7）とは別に、**実際の測定・$U$・補正の回路**を単一の配置入力 $|a\rangle|0\rangle_g$ に作用させて $L=3,4$ で照合する（ノイズなし最小版：$S^g_X$ のみ測定）。同じ測定分岐（$m$ と $Z^g$ の結果を固定）で 2 つの配置 $a_1,a_2$（異なる論理剰余類を含む）の出力振幅比が $\omega^{F(a_2)-F(a_1)}$ に一致すること＝回路が配置ごとに $F$ の位相を与えること、を確認する。プロトコルは線形なので、これは符号状態に対する作用を決める。"""),
    code(r"""def protocol_on_config(lat, d4, a, rng, forced_m=None, forced_z=None):
    psi = SparseState({frozenset(a): 1.0})
    ggens = [(lab, op) for lab, op in d4.xgens if lab[2] == 'g']
    m = []
    for i, (lab, op) in enumerate(ggens):
        bit, psi, p = psi.measure_op(op, rng, forced=None if forced_m is None else forced_m[i]); m.append(bit)
    psi = psi.apply_perm(lambda x: V_map(lat, x)).apply_diag(W_operator(lat))
    zb = []
    for j, e in enumerate(range(lat.E)):
        bit, psi, p = psi.measure_z(lat.q(e, G), rng, forced=None if forced_z is None else forced_z[j]); zb.append(bit)
        if p < 1e-12: return None, m, zb
    ag = {lat.q(e, G) for e, b in zip(range(lat.E), zb) if b}
    kappa = f2_solve(np.array([lat.vec(op.s) for _, op in ggens]), lat.vec(ag))
    for i in np.nonzero(kappa)[0]:
        psi = psi.apply_psop(ggens[i][1])
    for (lab, op), bit in zip(ggens, m):
        if bit:
            x0, y0 = lab[1]
            psi = psi.apply_diag(DiagPhase([(4, (lat.q(lat.eidx[('v', x0, y)], c),)) for y in range(y0) for c in (R, B)]))
    assert set(psi.a) == {frozenset(a)}, 'output not proportional to the input configuration'
    return psi.a[frozenset(a)], m, zb
for L, ntr in [(3, 6), (4, 2 if QUICK else 4)]:
    t0 = time.time()
    lat = D4Lattice(L); d4 = CodeData(lat); fsc = CodeData(lat, (R, B), dressed=False)
    gverts = [lab[1] for lab, op in d4.xgens if lab[2] == 'g']; Wop = W_operator(lat)
    def rand_cfg(coset):
        x = set()
        for i in np.nonzero(rng.integers(0, 2, len(fsc.xgens)))[0]: x ^= set(fsc.xgens[i][1].s)
        if coset: x ^= set(np.nonzero(fsc.t_log)[0])
        return frozenset(x)
    ok, nb = True, 0
    for _ in range(ntr):
        a1 = rand_cfg(0)
        A1, m, zb = protocol_on_config(lat, d4, a1, rng)
        for a2 in [rand_cfg(0), rand_cfg(1), rand_cfg(1)]:
            A2, _, _ = protocol_on_config(lat, d4, a2, rng, forced_m=m, forced_z=zb)
            if A2 is None: continue
            nb += 1
            pred = OMEGA ** ((protocol_phase(lat, Wop, a2, gverts) - protocol_phase(lat, Wop, a1, gverts)) % 8)
            ok &= abs(A2 / A1 - pred) < 1e-9
    record(f'V7b L={L}: protocol circuit on configuration inputs reproduces the phase formula F(a) (incl. T^dag between cosets)',
           ok and nb > 0, f'{nb} same-branch pairs, {time.time()-t0:.0f}s')"""),
    md(r'''## V8. 大きな $L$ でのプロトコルの正しさ（局所厳密証明）

V7 の手順を解析すると、FSC の配置 $a=(a_r,a_b)$ が受け取る位相は
$$F(a) = w(a) + 4\sum_{v} c'_v(a)\, f^g_v(a)\pmod 8,\qquad \delta c' = a_r + a_b,\; c'|_{\mathcal L_{rb}}=0$$
（$w$ は $W$ の位相指数、$f^g_v$ は $S^g_{X;v}$ の CZ$^{r,b}$ 部分）となる（$m$ に依存する部分は Pauli 補正で消え、$\kappa$ に依存する部分は $S^g$ 補正で消える）。この式は V7 の厳密シミュレーションで $L\le2$ について裏付けられている。
プロトコルが論理 $T^\dagger$ を実装する ⇔ $F$ が論理剰余類上で一定で $F(1)-F(0)=7$。
FSC の各生成元シフト $t$ について $F(a+t)=F(a)$ を、$F(a+t)-F(a)$ が依存する量子ビット（と $c'_{w_0}(a)$ という $\mathbb F_2$ 線形汎関数）への $C_{\rm FSC}$ の**厳密な射影**の全要素で確認する。'''),
    code(r'''for L in range(1, L_MAX_INVARIANCE + 1):
    t0 = time.time()
    lat = D4Lattice(L); fsc = CodeData(lat, (R, B), dressed=False); d4 = CodeData(lat)
    gverts = [lab[1] for lab, op in d4.xgens if lab[2] == 'g']; Wop = W_operator(lat)
    fails, mx = 0, 0
    for lab, op in fsc.xgens:
        ok, nloc = check_protocol_invariance(lat, fsc, Wop, gverts, op); fails += (not ok); mx = max(mx, nloc)
    F0 = protocol_phase(lat, Wop, frozenset(), gverts)
    F1 = protocol_phase(lat, Wop, frozenset(np.nonzero(fsc.t_log)[0]), gverts)
    record(f'V8 L={L}: F invariant under all {len(fsc.xgens)} FSC generators; F(1)-F(0)=7 (T^dag)',
           fails == 0 and (F1 - F0) % 8 == 7, f'max local configs {mx}, F0={F0}, F1={F1}, {time.time() - t0:.1f}s')'''),
    md(r'''## V9. 測定ガジェット（制御-$S$ ＋ $X$ 測定）の正しさ

$L=3$ の全 $X$ 型生成元（frozen を除かない版：最も多くの CCZ を含む）について、ガジェット回路をその台の上の密ベクトルで実行し、補助を $|\pm\rangle$ に射影した後のデータ状態が $(1\pm S)|\psi\rangle/2$ と一致することを確認する（$|\psi\rangle$ はランダム）。frozen 量子ビットを除いた版は、frozen を $|0\rangle$ とした状態で同じ検査を行う。'''),
    code(r'''def gadget_dense_check(P, lab, rng, frozen_zero=False):
    gad = P.x_gadget(lab, True)
    op = PSOp()
    for ch in lab[2]:
        op = op @ P.lat.SX(lab[1], CNAME.index(ch))
    anc = gad[0][1]
    qs = sorted((gadget_support(gad) | op.support()) - {anc}) + [anc]
    k = len(qs); pos = {q: i for i, q in enumerate(qs)}; idx = np.arange(2 ** k)
    b = lambda q: (idx >> (k - 1 - pos[q])) & 1
    psi = rng.normal(size=2 ** k) + 1j * rng.normal(size=2 ** k)
    psi = np.where(b(anc) == 0, psi, 0)
    if frozen_zero:
        for q in qs:
            if q in P.frozen: psi = np.where(b(q) == 0, psi, 0)
    psi /= np.linalg.norm(psi)
    st = (psi + psi[idx ^ (1 << (k - 1 - pos[anc]))]) / np.sqrt(2)          # ancilla |0> -> |+>
    for g in gad[1:-1]:
        if g[0] == 'CNOT':
            st = np.where(b(g[1]) == 1, st[idx ^ (1 << (k - 1 - pos[g[2]]))], st)
        elif g[0] == 'CCZ':
            st = np.where(b(g[1]) & b(g[2]) & b(g[3]), -st, st)
        else:
            raise ValueError(g)
    Spsi = np.zeros_like(psi)
    for i in np.nonzero(np.abs(psi) > 0)[0]:
        x = frozenset(q for q in qs if q != anc and (i >> (k - 1 - pos[q])) & 1)
        y, sb = op.apply(x)
        j = sum(1 << (k - 1 - pos[q]) for q in y)
        Spsi[j] += (-1) ** sb * psi[i]
    ok = True
    for sgn in (1, -1):
        a0 = np.where(b(anc) == 0, st, 0); a1 = np.where(b(anc) == 1, st, 0)
        post = (a0 + sgn * a1[idx ^ (1 << (k - 1 - pos[anc]))]) / np.sqrt(2)
        ok &= np.allclose(post, (psi + sgn * Spsi) / 2)
    return ok
Pu = D4Protocol(3, prune_frozen=False); Pp = D4Protocol(3, prune_frozen=True)
okU = all(gadget_dense_check(Pu, lab, rng) for lab, _ in Pu.d4.xgens)
okP = all(gadget_dense_check(Pp, lab, rng, frozen_zero=True) for lab, _ in Pp.d4.xgens)
record('V9 L=3: every X-type gadget (unpruned) implements the projective measurement of its Clifford stabilizer', okU)
record('V9 L=3: pruned gadgets (frozen qubits removed) agree when frozen qubits are |0>', okP)'''),
    md(r'''## V10. スケジュールの妥当性（独立な行列計算）

可換クラス・スケジュールの上界は「同じクラスの異なるガジェットのゲートは全て可換」という条件に依存する（その場合、同時実行は逐次測定と等価）。役割による判定とは独立に、共有量子ビットを持つゲート対の行列（≤6量子ビット）を作って交換子を計算する。$U$ のゲート同士の可換性も同様に確認する（$U$ を1層に詰めてよい根拠）。'''),
    code(r'''GMAT = {'CNOT': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex),
        'CZ': np.diag([1, 1, 1, -1]).astype(complex), 'CS': np.diag([1, 1, 1, 1j]), 'CSDG': np.diag([1, 1, 1, -1j]),
        'CCZ': np.diag([1] * 7 + [-1]).astype(complex), 'T': np.diag([1, np.exp(1j * np.pi / 4)]),
        'TDG': np.diag([1, np.exp(-1j * np.pi / 4)])}
def embed(g, qs):
    k = len(qs); pos = {q: i for i, q in enumerate(qs)}; G_ = GMAT[g[0]]; a = len(g) - 1
    M = np.zeros((2 ** k, 2 ** k), complex)
    for col in range(2 ** k):
        bits = [(col >> (k - 1 - i)) & 1 for i in range(k)]
        sub = sum(bits[pos[q]] << (a - 1 - t) for t, q in enumerate(g[1:]))
        for out in range(2 ** a):
            amp = G_[out, sub]
            if amp == 0: continue
            nb = bits[:]
            for t, q in enumerate(g[1:]): nb[pos[q]] = (out >> (a - 1 - t)) & 1
            M[sum(bb << (k - 1 - i) for i, bb in enumerate(nb)), col] += amp
    return M
_cache = {}
def commute(g1, g2):
    key = (g1, g2)
    if key not in _cache:
        qs = sorted(set(g1[1:]) | set(g2[1:])); A, Bm_ = embed(g1, qs), embed(g2, qs)
        _cache[key] = np.allclose(A @ Bm_, Bm_ @ A)
    return _cache[key]
P3 = D4Protocol(3)
allok, npairs = True, 0
for code_ in ['D4', 'FSC']:
    for cls_ in commuting_classes(P3.round_gadgets(code_)):
        ents = [[g for g in gad if g[0] in GMAT] for gad in cls_]
        byq = defaultdict(list)
        for gi, gl in enumerate(ents):
            for g in gl:
                for q in g[1:]: byq[q].append((gi, g))
        for q, lst in byq.items():
            for (i1, g1), (i2, g2) in itertools.combinations(lst, 2):
                if i1 != i2:
                    npairs += 1
                    if not commute(g1, g2): allok = False
record('V10 L=3: all gates of different gadgets within each commuting class commute (matrix check)', allok, f'{npairs} gate pairs')
Ug = P3.U_gates(); okUc = True
for g1, g2 in itertools.combinations(Ug, 2):
    if set(g1[1:]) & set(g2[1:]) and not commute(g1, g2): okUc = False
record('V10 L=3: all gates of U = W V mutually commute (U can be layered freely)', okUc, f'{len(Ug)} gates')
# Clifford+T decompositions used in the 'clifford_t' gate-set model
def circuit_matrix(gates, k):
    M = np.eye(2 ** k, dtype=complex)
    for g in gates: M = embed(g, list(range(k))) @ M
    return M
record('V10 CCZ = 7-T decomposition (matrix)', np.allclose(circuit_matrix(ccz_to_clifford_t(0, 1, 2), 3), GMAT['CCZ']))
record('V10 CS = 3-T decomposition (matrix)', np.allclose(circuit_matrix(cs_to_clifford_t(0, 1), 2), GMAT['CS']))
record('V10 CS^dag decomposition (matrix)', np.allclose(circuit_matrix(csdg_to_clifford_t(0, 1), 2), GMAT['CSDG']))'''),
    md(r'''## V11. 資源計数の独立照合（幾何からの直接公式）
格子の幾何から手で導いた式と、ガジェット生成コードの計数を比較する。
* データ（frozen 除く）：辺数 $2L(L+1)$ × 3色 − frozen（$\mathcal L_r$ の $r$: $L$、$\mathcal L_b$ の $b$: $L$、$\mathcal L_{rb}$ の $g$: $2L$）$=6L^2+2L$。
* $X$ 型生成元：バルク $3(L-1)^2$、$\mathcal L_r$・$\mathcal L_b$ 各 $2(L-1)$、左上角1、$\mathcal L_{rb}$ $2L-1$ → 合計 $3L^2-2L+1$… などを直接列挙と比較。
* CCZ／ラウンド（frozen 除去後）：CZ 対のうち両端が非 frozen のもの。'''),
    code(r'''ok = True; rows = []
for L in range(2, 10):
    P = D4Protocol(L); qs = P.qubit_sets()
    E = 2 * L * (L + 1)
    data_formula = 3 * E - (L + L + 2 * L)
    xg_formula = 3 * (L - 1) ** 2 + 2 * (L - 1) + 2 * (L - 1) + 1 + (2 * L - 1)
    zc_formula = 3 * L * L + 2 * L        # plaquettes (3 colours) + Z^rZ^b on the 2L L_rb edges
    ccz_direct = sum(1 for lab, _ in P.d4.xgens for ch in lab[2]
                     for p in P.lat.cz_pairs(lab[1], CNAME.index(ch)) if p[0] not in P.frozen and p[1] not in P.frozen)
    ccz_code = sum(1 for gd in P.round_gadgets('D4') for g in gd if g[0] == 'CCZ')
    r = dict(L=L, data=len(qs['d4_data']), data_formula=data_formula, xgens=len(P.d4.xgens), xg_formula=xg_formula,
             anc=len(qs['d4_anc']), anc_formula=xg_formula + zc_formula, ccz=ccz_code, ccz_direct=ccz_direct,
             ccz_closed_form=6 * (L - 1) ** 2)
    rows.append(r)
    ok &= r['data'] == data_formula and r['xgens'] == xg_formula and r['anc'] == r['anc_formula'] and ccz_code == ccz_direct == r['ccz_closed_form']
record('V11 qubit/generator/CCZ counts agree with direct geometric formulas (L=2..9)', ok)
pd.DataFrame(rows)'''),
    md(r'''# 第 2 部 15-to-1 と GoSC の配置（雑音なし、v4 から引き継ぎ）'''),
    md(r'''## M1. 行列と回路の整合
Fig. 15 の各列（qubit 1..5 の作用）を図から手で転記した表 `FIG15_TRANSCRIBED` と、式(1) から機械的に生成した回転を比較する。'''),
    code(r'''FIG15_TRANSCRIBED = ['11ZZZ', '1Z1ZZ', '1ZZ1Z', '1ZZZ1', 'Z11ZZ', 'Z1Z1Z', 'Z1ZZ1', 'ZZ11Z', 'ZZ1Z1', 'ZZZ11', 'ZZZZZ']
derived = [''.join('Z' if i in r else '1' for i in range(5)) for r in fig15_rotations()]
record('M1a triorthogonality of G (eq. 2)', all(triorthogonal(G_TRI)), triorthogonal(G_TRI))
# G~ is not literally row-equivalent to G (the row spaces differ): the elimination in [Lit] also reorders
# columns.  Equivalence up to row operations + column permutation: the 16 columns of G are exactly the affine
# hyperplane {c : c_0 = 1}; we check that the 16 columns of G~ are exactly an affine hyperplane {c : f.c = 1}.
def affine_hyperplane_functional(Mx):
    for f in itertools.product((0, 1), repeat=Mx.shape[0]):
        if len({tuple(c) for c in Mx.T}) == 2 ** (Mx.shape[0] - 1) and all(np.dot(f, c) % 2 == 1 for c in Mx.T):
            return f
    return None
fG, fGt = affine_hyperplane_functional(G_TRI), affine_hyperplane_functional(G_TILDE)
record('M1b G~ (eq. 4) is triorthogonal and equivalent to G (eq. 2) up to row operations and a column permutation',
       all(triorthogonal(G_TILDE)) and fG is not None and fGt is not None,
       f'columns of G: hyperplane f={fG}; columns of G~: hyperplane f={fGt}; literal row-equivalence: '
       f'{f2_rank(np.vstack([G_TRI, G_TILDE])) == 5} (column order differs)')
record('M1c M_15-to-1 (eq. 1) = G~ with the first column removed (puncturing)', (G_TILDE[:, 1:] == M15).all())
record('M1d 11 rotations derived from eq. (1) = Fig. 15 (manual transcription)', derived == FIG15_TRANSCRIBED, derived)'''),
    md(r'''## M2. 理想入力での 5 量子ビット回路'''),
    code(r'''res = run_15to1_5qubit()
nz = {k: v for k, v in res.items() if v[0] > 1e-12}
Tp, Tdp = T_GATE @ KETP, T_GATE.conj() @ KETP
out = nz.get((1, 1, 1, 1), (0, None))[1]
record('M2a ideal inputs: X outcomes of qubits 1-4 are deterministically +1', list(nz) == [(1, 1, 1, 1)] and abs(nz[(1, 1, 1, 1)][0] - 1) < 1e-12)
record('M2b output of qubit 5 = T^dag|+> (not T|+>)', equal_up_to_phase(out, Tdp) and not equal_up_to_phase(out, Tp))
record('M2c X T^dag|+> = e^{-i pi/4} T|+>  (Pauli-frame unification used for both protocols)',
       np.allclose(X @ Tdp, np.exp(-1j * np.pi / 4) * Tp))'''),
    md(r'''## M3. 15 量子ビット符号（Fig. 14）'''),
    code(r'''rT = rm15_logical_check('T'); rTd = rm15_logical_check('Tdag')
record('M3a transversal T on |+_L> gives <X_L>=1/sqrt2, <Y_L>=-1/sqrt2 (logical T^dag|+>), Z stabilisers kept',
       abs(rT['X_L'] - 2 ** -.5) < 1e-9 and abs(rT['Y_L'] + 2 ** -.5) < 1e-9 and rT['min_Zstab'] > 1 - 1e-9, rT)
record('M3b transversal T^dag gives logical T|+>', abs(rTd['Y_L'] - 2 ** -.5) < 1e-9, rTd)
bloch_5q = [np.real(np.vdot(out, P @ out)) for P in (X, Y, Z)]
record('M3c 5-qubit circuit output has the same Bloch vector as the 15-qubit logical state',
       np.allclose(bloch_5q, [rT['X_L'], rT['Y_L'], rT['Z_L']], atol=1e-9), np.round(bloch_5q, 6))'''),
    md(r'''## M4. 誤り検出（15 個の π/8 回転のどれかに $Z$ 誤り）'''),
    code(r'''cols15 = [tuple(int(i) for i in np.nonzero(M15[:, j])[0]) for j in range(15)]
def accept(err):
    psi = kron_all([KETP.reshape(2, 1)] * 5).reshape(-1)
    for j, qs in enumerate(cols15):
        psi = z_rotation_diag(5, qs, np.pi / 8 + (np.pi / 2 if j in err else 0)) * psi
    p = 1.0; st = psi
    for q in range(4):
        _, st, pp = measure_pauli(st, pauli_on(5, {q: 'X'}), forced=1); p *= pp
        if pp < 1e-14: return 0.0, None
    o = kron_all([np.array([1, 1]) / np.sqrt(2)] * 4).reshape(-1).conj() @ st.reshape(16, 2)
    return p, o / np.linalg.norm(o)
und = {}
ideal = accept(())[1]
wrong3 = 0
for w in (1, 2, 3):
    und[w] = 0
    for e in itertools.combinations(range(15), w):
        p, o = accept(e)
        if p > 1e-9:
            und[w] += 1
            wrong3 += (w == 3 and not equal_up_to_phase(o, ideal))
record('M4 all 1- and 2-error patterns detected; 35 undetected 3-error patterns, all with a wrong output (output error 35 p^3)',
       und[1] == 0 and und[2] == 0 and und[3] == 35 and wrong3 == 35, f'undetected {und}; wrong outputs among undetected weight-3: {wrong3}')'''),
    md(r'''## M5. selective π/4（Fig. 17a）と auto-corrected π/8（Fig. 17b）
選んだ 4 種の Pauli 積 $P$（1〜3 量子ビット）とランダムな入力状態について、**全ての測定結果の分岐**で、補正後のデータ状態が目的の回転と一致するか（必要な補正が Pauli $P$ だけか）を確認する。任意の $P$・任意の入力についての証明ではない（線形性により入力は 1 つのランダム状態で十分だが、$P$ は代表例のみ）。'''),
    code(r'''def selective_pi4(P_data, psi, basis):
    k = int(np.log2(len(psi))); n = k + 1; a = k
    st0 = np.kron(psi, KET0)
    Pd = pauli_on(k, P_data)
    target = (np.cos(np.pi / 4) * psi - 1j * np.sin(np.pi / 4) * (Pd @ psi)) if basis == 'X' else psi
    outs = []
    for o1 in (1, -1):
        _, st, p1 = measure_pauli(st0, pauli_on(n, {**P_data, a: 'Y'}), forced=o1)
        if p1 < 1e-14: continue
        for o2 in (1, -1):
            _, st2, p2 = measure_pauli(st, pauli_on(n, {a: basis}), forced=o2)
            if p2 < 1e-14: continue
            anc = (np.array([1, o2]) / np.sqrt(2)) if basis == 'X' else (KET0 if o2 == 1 else np.array([0, 1]))
            dat = st2.reshape(2 ** k, 2) @ anc.conj(); dat /= np.linalg.norm(dat)
            outs.append('I' if equal_up_to_phase(dat, target) else 'P' if equal_up_to_phase(Pd @ dat, target) else 'FAIL')
    return outs
ok = True; table = []
for P in [{0: 'Z'}, {0: 'Z', 1: 'Z'}, {0: 'X', 1: 'Y'}, {0: 'Y', 1: 'Z', 2: 'X'}]:
    k = max(P) + 1
    psi = rng.normal(size=2 ** k) + 1j * rng.normal(size=2 ** k); psi /= np.linalg.norm(psi)
    for basis in 'XZ':
        o = selective_pi4(P, psi, basis); ok &= 'FAIL' not in o
    br = auto_corrected_pi8(P, psi)
    ok &= all(b['correction'] != 'FAIL' for b in br) and abs(sum(b['prob'] for b in br) - 1) < 1e-9
    table += [dict(P=str(P), **{kk: b[kk] for kk in ['PZ', 'ZY', 'anc_basis', 'anc', 'Xm', 'correction']}, prob=round(b['prob'], 4)) for b in br]
record('M5 selective pi/4 and auto-corrected pi/8 (4 sample Pauli products, random input): every branch correct up to a Pauli P correction', ok)
pd.DataFrame(table).head(16)'''),
    md(r'''## M6. tile 数・time step の再計数
(a) 格子転記 `FIG17C_*` から数える。(b) [Lit] 本文の記述（「4 qubits + 1 output」「左の 2×2 ブロックは $|m\rangle$ と $|0\rangle$ 用」「2 tiles は多体測定のアンシラ」「11 回転 × 1τ」）から独立に数える。'''),
    code(r'''sched = tile_schedule(); n_a = len(tiles_used(sched)); tau_a = sum(t for _, t, _ in sched)
n_b = 4 + 1 + 2 * 2 + 2; tau_b = 11 * 1
record('M6 tiles = 11 and time steps = 11 (two independent countings), space-time = 121 d^3',
       n_a == n_b == 11 and tau_a == tau_b == 11, dict(grid=(n_a, tau_a), text=(n_b, tau_b)))
for dd in [3, 5, 7, 9]:
    s = summarize(msd_stages(MSDCostData(dd), 'FT_ref', DEFAULT_PARAMS, 'A', level='tile', data_only=True))
    s2 = summarize(msd_stages(MSDCostData(dd), 'FT_ref', DEFAULT_PARAMS, 'A', level='tile', data_only=False))
    record(f'M6 d={dd}: tile model reproduces 121 d^3 (data) and 242 d^3 (incl. measurement qubits)',
           s['V_qubit_round'] == 121 * dd ** 3 and s2['V_qubit_round'] == 242 * dd ** 3)'''),
    md(r'''## M7. 物理レベルの格子手術（stim、ノイズなし）
2 つの $d\times d$ 回転表面符号パッチを $|+_L\rangle$ に準備し、間の 1 tile（幅 $d$）を $|+\rangle$ で初期化して合併矩形の安定化子を 1 ラウンド測定する。$\bar Z_1\bar Z_2$ の期待値が合併前 0 → 合併後 ±1 となり、その符号が合併で新たに測った $Z$ 型チェックの積と一致することを確認する（[Lit] App. A の手順）。'''),
    code(r'''for dd in [3, 5, 7]:
    for seed in range(3):
        r = lattice_surgery_zz_check(dd, seed=seed)
        record(f'M7 d={dd} seed={seed}: Z1Z2 random before, deterministic after merge, equals product of merged Z checks',
               r['exp_before'] == 0 and abs(r['exp_after']) == 1 and r['consistent'] and r['O_in_span'],
               f"after={r['exp_after']}, qubits={r['total_qubits']}")
    md_ = MSDCostData(dd)
    record(f'M7 d={dd}: one gap tile adds 2d^2+1 qubits to the merged region (used for the ancilla strip)', md_.gap_tile == 2 * dd * dd + 1, md_.gap_tile)'''),
    md(r'''## M8. パッチの量子ビット数・CNOT 数と換算の一貫性'''),
    code(r'''ok = True
for dd in range(3, 16, 2):
    m = MSDCostData(dd)
    ok &= m.sq_patch == 2 * dd * dd - 1 and m.tall_patch == 4 * dd * dd - 1 and m.cnot_sq == 4 * dd * (dd - 1)
    phys = summarize(msd_stages(m, 'FT_ref', DEFAULT_PARAMS, 'A', level='physical'))
    ok &= phys['peak_qubits'] == 22 * dd * dd - 6 and phys['V_qubit_round'] == 11 * dd * (22 * dd * dd - 6)
record('M8 rotated patch = 2d^2-1 qubits, 4d(d-1) CNOT/round; explicit physical model: 22d^2-6 qubits per rotation (tile model 22d^2), V = 11 d (22d^2-6)', ok)
m = MSDCostData(5)
record('M8 surface-code round time bounds from the shared scheduler (lb=6: init+4 CNOT+meas)', m.round['time_lb'] == 6, m.round)'''),
    md(r"""## M9. Fig. 17c の 11 個の多パッチ測定の物理レベル構成と検証

各回転 $k$ で測る演算子は $O_k=\bigotimes_{i\in P_k}\bar Z_i\otimes\bar Z_m$（$P_k$ は式(1)の列）。補助帯を $|+\rangle$ に初期化し、新しい検査 $N$ を測ったときに：
* V1 $N$ が互いに可換、V2 各パッチが単独で $k=1$・距離 $d$、
* V3 $O_k\in\mathrm{span}(N_Z)+\mathrm{span}(G_{0,Z})$ で、測られる $Z$ 論理の組合せはちょうど 1 個、V4 $X$ 論理は 1 個も測られない、
* V5 $O_k$ と可換な全ての論理演算子（全 $\bar Z_i$、非参加パッチの $\bar X_j$、参加パッチ対の $\bar X_i\bar X_{i'}$）が $N$ と可換な代表元をもつ（保存される）、
* 合併符号 $G_1=\langle N,\ G_0\text{ の }N\text{ 可換部分群}\rangle$ の論理量子ビット数が 5、距離 $\ge d$（MILP）、

を厳密に確認する。さらに stim で、合併前は $O_k$ がランダム、合併 1 ラウンド後は決定的で、その値が測定結果の積から予言どおりであることを確認する。"""),
    code(r"""import time
rot_sets = [tuple(sorted({'q%d' % (i + 1) for i in r})) + ('m',) for r in fig15_rotations()]
for dd in [3, 5, 7]:
    t0 = time.time(); allv = True; dists = set()
    for P in rot_sets:
        r = verify_multipatch(dd, P, distance=(dd <= 5))
        allv &= r['valid'] and r['V2_patches_k1']
        if dd <= 5:
            allv &= r['V2_patch_distance_ok']; dists.add(r['V6_merged_distance'])
    record(f'M9 d={dd}: all 11 multi-patch measurements P(x)Z_m are valid lattice surgeries (exact F2 criteria'
           + (', merged-code distance >= d' if dd <= 5 else '') + ')', allv,
           (f'merged (d_X, d_Z) = {sorted(dists)}; ' if dists else 'distance not computed at this d; ') + f'{time.time() - t0:.0f}s')
for dd in [3, 5]:
    ok = all(stim_multipatch_check(dd, P, seed=s)['consistent'] for P in rot_sets for s in range(2))
    record(f'M9 d={dd}: stim - O_k random before, deterministic after one merged round, equal to predicted product (11 rotations x 2 seeds)', ok)
ok = True
for dd in [3, 5, 7]:
    for P in rot_sets:
        a, res = rotation_measured_checks(dd, P)
        b, pats, own = rotation_measured_checks_fast(dd, P)
        qs = sorted(own); idx = {q: i for i, q in enumerate(qs)}; n = len(qs)
        for t in 'XZ':
            A = np.array([_vec(c[1], idx, n) for c in a if c[0] == t]); Bm = np.array([_vec(c[1], idx, n) for c in b if c[0] == t])
            ok &= _rank(A) == _rank(Bm) == _rank(np.vstack([A, Bm]))
        ok &= len(a) == len(b) and max(len(c[1]) for c in b) <= 4
record('M9 fast measured-set construction (used for d up to 15) = verified construction (d=3,5,7); all checks weight <= 4', ok)"""),
    md(r'''# 第 3 部 雑音付きシミュレーションの検証（v5）
## N1. Takada らの方法の適用条件
Takada–Bartlett–Williamson の Clifford-stabilizer simulation（Pauli proxy 法）は、非 Clifford な「Clifford 安定化子」を測る回路に一般に使えるわけではない。十分条件（Definition 2 と Sec. VIII）を D4 code switching の各段について確認する（`takada_conditions`）：
* (a) X 型 Clifford 安定化子の X 因子と CZ 因子の台が互いに素（transversal、式 (131)）。
* (b) CZ, S への X/Y 誤りのドレッシングが単一量子ビット $Z$（ドレッシングの軸が全量子ビットで $Z$）。行列は**大域位相だけを許す厳密比較**（`equal_up_to_phase`）で比べる（以前の版は要素の絶対値を比べており、$I$ と $Z$ を区別できなかった。下の負の対照で、この判定が $I$・別の軸・相対位相の違いを FAIL にすることを確認する）。
* (c) 測定する生成元どうしの群交換子が、その時点の $Z$ 安定化子の張る空間に入る（実効的な可換性、式 (58) / Lemma 1）。switch-in では FSC $|+_L\rangle\otimes|0\rangle_g$ を、D4 側の CZ でドレスした生成元とドレスした論理 $\bar X$ で表して確認する（Sec. VIII B の生成元の選び方）。
* (d) primary-support exactness（Definition 1）：X 台が線形独立。
* (e) $U$ と switch-out の補正網が CNOT と対角第 3 階層ゲートのみ（Sec. VIII A, 式 (178)）。
* (f) 出力の論理安定化子 $\bar H$ が Clifford で、FSC の X 検査と実効的に可換。

**範囲**：これらは各段で測る生成元・論理演算子についての有限サイズ（$L=2..7$, 9；本番は $d=3,5,7,9$）の代数的確認である。プロトコル全体（切り替え・$U$・switch-out の順序を含む）で proxy が真の回路と一致することは、別に N3 で数値的に確認する（$L\le2$）。一般の $L$ についての証明ではない。'''),
    code(r'''I2_, Z_, S_ = np.eye(2), np.diag([1, -1]), np.diag([1, 1j])
neg = [equal_up_to_phase(np.eye(4), np.kron(I2_, Z_)), equal_up_to_phase(S_, Z_),
       equal_up_to_phase(np.kron(Z_, I2_), np.kron(I2_, Z_)), equal_up_to_phase(np.diag([1, 1j]), np.diag([1, -1j]))]
pos = [equal_up_to_phase(1j * Z_, Z_), equal_up_to_phase(-np.kron(I2_, Z_), np.kron(I2_, Z_))]
record('N1 operator comparison is exact up to a global phase (I vs Z, S vs Z, wrong qubit, relative phase rejected)',
       not any(neg) and all(pos), f'negative controls {neg}, positive controls {pos}')
for L in list(range(2, 8)) + ([] if QUICK else [9]):
    t0 = time.time()
    for name, (ok, det) in takada_conditions(L).items():
        record(f'N1 L={L}: {name}', ok, det)
    print('   (%.0f s)' % (time.time() - t0))'''),
    md(r'''## N2. ドレッシングの代数
誤りの X 成分 $X^e$ が対角ゲート $D$（CCZ, CS, CS$^\dagger$, $T$, $T^\dagger$）を通過するとき、Pauli proxy 回路は $D$ を取り除き、$C_e=D^\dagger X^e D X^e$ を挿入する。$C_e$ が S, Z, CZ の積（Clifford）であり、`dressing()` が返すゲート列と大域位相を除いて一致することを全ての $e$ について行列で確かめる。'''),
    code(r'''def _kron(ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out
_I2 = np.eye(2, dtype=complex); _X = np.array([[0, 1], [1, 0]], dtype=complex)
_G1 = {'S': np.diag([1, 1j]), 'SDG': np.diag([1, -1j]), 'Z': np.diag([1, -1]).astype(complex)}
ok, n = True, 0
for name, k in [('CCZ', 3), ('NCCZ', 3), ('CS', 2), ('CSDG', 2), ('T', 1), ('TDG', 1)]:
    D = np.diag([np.exp(1j * np.pi / 4 * DIAG_THETA[name](x)) for x in itertools.product((0, 1), repeat=k)])
    for e in itertools.product((0, 1), repeat=k):
        Xe = _kron([_X if b else _I2 for b in e])
        Ce = D.conj().T @ Xe @ D @ Xe
        M = np.eye(2 ** k, dtype=complex)
        for g in dressing(name, e):
            if g[0] == 'CZ':
                M = np.diag([(-1) ** (x[g[1]] * x[g[2]]) for x in itertools.product((0, 1), repeat=k)]).astype(complex) @ M
            else:
                M = _kron([_G1[g[0]] if i == g[1] else _I2 for i in range(k)]) @ M
        ph = Ce[0, 0] / M[0, 0]
        ok &= abs(abs(ph) - 1) < 1e-12 and np.allclose(Ce, ph * M)
        n += 1
record('N2 dressing C_e = D^dag X^e D X^e equals the inserted S/Z/CZ gates (all diagonal gates, all e)', ok, f'{n} cases')'''),
    md(r'''## N3. Pauli proxy 回路と真の非 Clifford 回路の並走（最も重要な検査）
同じ誤り（雑音モデルから標本）・同じ測定結果・同じランダムな $X$ 補正を、(i) Pauli proxy 回路（stim、本番と同じ `FastShot`）と (ii) 真の非 Clifford 回路の厳密な疎状態ベクトル（`TrueSim`、CCZ・CS・$T^\dagger$ をそのまま作用）に与え、
* proxy で決定的な測定は、真の回路でも確率 1 で同じ値になる；proxy でランダムな測定は、真の回路でも確率 1/2；
* 最後の論理 Clifford 安定化子の期待値 $\langle\bar H\rangle$ が一致する
ことを全測定について確認する。Takada らの定理（proxy が真の回路と同じ測定統計を与える）を、この回路について数値的に直接確かめるもの。$L=2$ は D4 相のラウンドが並列スケジュール（本番）と逐次スケジュールの両方。'''),
    code(r'''rows = []
for L, Rin, Rout, seq, p, ntr in [(1, 2, 1, False, 0.03, 30), (2, 2, 1, False, 0.01, 4 if QUICK else 8),
                                  (2, 2, 1, True, 0.01, 2 if QUICK else 4), (2, 2, 1, False, 0.0, 1)]:
    t0 = time.time(); bad = 0; nm = nr = 0; hs = []
    for tr in range(ntr):
        r = lockstep_validation(L, Rin, Rout, NoiseModel(p), rng, sequential=seq, corr_prob=0.05 if p else 0)
        bad += not r['agree']; nm += r['n_meas']; nr += r['n_random']; hs.append(round(float(np.real(r['H_true'])), 6))
        rows.append(dict(L=L, sequential=seq, p=p, trial=tr, n_meas=r['n_meas'], n_random=r['n_random'],
                         n_mismatch=r['n_mismatch'], H_true=np.real(r['H_true']), H_proxy=r['H_proxy'], n_dress=r['n_dress']))
    record(f'N3 L={L} {"sequential" if seq else "parallel"} schedule, p={p}: proxy = exact non-Clifford circuit ({ntr} runs)',
           bad == 0 and (p > 0 or hs == [1.0]), f'{nm} measurements ({nr} random), <H> values {sorted(set(hs))}, {time.time() - t0:.0f}s')
pd.DataFrame(rows).groupby(['L', 'sequential', 'p']).agg(runs=('trial', 'count'), measurements=('n_meas', 'sum'),
                                                          mismatches=('n_mismatch', 'sum'), dressings=('n_dress', 'sum'))'''),
    md(r'''### N3b. 本番と同じ適応処理の全経路（$L=2$）
`lockstep_full`：本番の `D4Experiment.shot` をそのまま使い、flux 復号器の補正、switch-out の復号済み読み出し、物理的な補正 CZ の層とその雑音、Pauli frame、charge 復号器までを、proxy（stim）と真の回路の状態ベクトル（`TrueSim`）に同じ誤り・同じ測定結果で並走させる。S0（Clifford 回路で stim だけで厳密）は、その理想出力 $E|+_L\rangle|0\rangle_g$（$E$ はランダムな 2 量子ビット Pauli）で置き換える（S0 の多数の補助量子ビットを含む状態ベクトルはメモリに入らない）。真の回路側は proxy 側の状態や誤りの表現を使わず、ゲートを状態ベクトルに直接作用させる。'''),
    code(r'''rows = []
for Rin, p, ntr in [(2, 0.01, 2 if QUICK else 6), (2, 0.03, 2 if QUICK else 6), (1, 0.001, 2)]:
    ex = D4Experiment(2, 1, Rin, 1, NoiseModel(p))
    t0 = time.time(); bad = nm = nr = ncz = nco = 0
    for tr in range(ntr):
        r = lockstep_full(2, Rin, 1, NoiseModel(p), rng, ex=ex)
        bad += not r['agree']; nm += r['n_meas']; nr += r['n_random']; ncz += r['so_cz']; nco += r['n_corr']
        rows.append(dict(Rin=Rin, p=p, trial=tr, **r))
    record(f'N3b L=2 Rin={Rin}, p={p}: full adaptive path (flux corrections, decoded switch-out, physical CZ layers '
           f'with noise, Pauli frame) proxy = exact non-Clifford circuit ({ntr} runs)', bad == 0,
           f'{nm} measurements ({nr} random), {nco} flux corrections, {ncz} physical switch-out CZs, {time.time() - t0:.0f}s')
pd.DataFrame(rows).groupby(['Rin', 'p']).agg(runs=('trial', 'count'), measurements=('n_meas', 'sum'),
                                             mismatches=('n_mismatch', 'sum'), corrections=('n_corr', 'sum'),
                                             switch_out_cz=('so_cz', 'sum'))'''),
    md(r'''## N4. 雑音なしの全体（復号器込み）
雑音を 0 にした本番と同じ実験（`D4Experiment`、復号器込み）で出力の不忠実度が 0 になること。switch-in のランダムな $S^g$ 測定結果、switch-out の $Z^g$ 読み出しとその補正、Pauli frame の扱いが正しいことを $d=3,5,7$ で確認する。'''),
    code(r'''for d in (3, 5, 7):
    ex = D4Experiment(d, d, d, d, NoiseModel(0.0))
    inf, st = ex.run(5, rng)
    record(f'N4 d={d}: noiseless protocol (with decoders) outputs the exact magic state', bool(np.all(inf == 0)),
           f'infidelities {inf.tolist()}')'''),
    md(r'''## N5. 1 個の故障（INFO：故障の網羅範囲と失敗の記録。合否ではなく、耐故障性を保証するものでもない）
距離 $d\ge3$ で耐故障な回路と復号器なら、1 個の故障は必ず訂正される。本研究の回路・復号器（[Dav] の JIT decoder は参照できず独自に構成）で、1 個の故障を強制し（他の雑音は 0）出力が誤るかを調べる（`single_fault_scan`）。
* **網羅範囲**：コンパイルした回路の全ての雑音位置（ゲート、CCZ・CS・$T^\dagger$、初期化、測定、アイドル）× その雑音チャネルの全ての Pauli、さらに switch-out の**動的な故障**（測定結果に応じて適用される各補正 CZ の後の 15 通りの Pauli、補正 CZ 層のアイドル）。古典処理は雑音なし。
* **測定分岐**：各故障を 2 つの測定乱数の種で実行する（ランダムな測定結果、したがって分岐が変わる）。どちらかで誤れば「失敗あり」。失敗しなかった故障が**全分岐で**訂正されるとは言えない。
* **一次の係数**：$c_1=\sum_f (P(f)/p)\,\overline{\varepsilon}(f)$（$P(f)$ はその故障の確率：1 量子ビット脱分極の各 Pauli $p/3$、2 量子ビット $p/15$、3 量子ビット $p/63$、反転 $p$、アイドルは $p_{\rm idle}/3$；$\overline\varepsilon$ は 2 つの種の不忠実度の平均）。小さい $p$ で $\varepsilon_K\approx c_1p$（この復号器での単一故障の寄与）。件数の比とは異なる。
* 窓（`D4Experiment.window`：D4 相最後のラウンド〜FSC の最初のラウンド）の内外に分けて記録する。窓の外の失敗（例えば D4 相のより前のラウンド）も数える。$d=3$ は全列挙、$d=5,7$ は失敗がまれなので無作為抽出では $c_1$ を推定できず、**窓の中を全列挙**する（$d=5$：A, B、$d=7$：B）。窓の外の寄与は $d=3$ の全列挙と $d=5$ の抽出で小さいことを確認する。$c_1$ の $d$ 依存は、比較ノートの「窓の項が $d$ によらない」という仮定の確認になる。
失敗の原因は、少なくとも (i) D4 相最後のラウンドの CCZ ガジェット内の故障（CZ のドレッシングによる非 Pauli な効果を charge 復号器の誤りモデルが含まない）、(ii) switch-out の時間境界での検出器の基底（どの基底でも一部の誤りがハイパーエッジ）、(iii) $U$ 直前の $Z$ ラウンドの後に起きて $U$ を通る $X$ 誤り（その時点で検出できない）を含む。原因を復号器・測定ガジェット・順序・境界のどれに帰すべきかの完全な分類はしていない（未解決）。'''),
    code(r'''t0 = time.time()
N5 = {}
CONFIGS = [(3, 1.0, None, False), (3, 0.0, None, False), (5, 1.0, 1500 if QUICK else 6000, False)]
if not QUICK:                                    # exhaustive over the switching window (where c1 sits)
    CONFIGS += [(5, 1.0, None, True), (5, 0.0, None, True), (7, 0.0, None, True)]
rows = []
for d, pi, ns, win in CONFIGS:
    ex = D4Experiment(d, d, d, d, NoiseModel(1e-3, p_idle=pi * 1e-3))
    r = single_fault_scan(ex, n_sample=ns, seeds=2, only_window=win)
    N5[(d, pi, win)] = r
    scope = 'switching window, all' if win else ('all' if ns is None else 'random sample of')
    info(f'N5 d={d}, scenario {"A" if pi else "B"}: single faults ({scope} {r["n_tested"]} of {r["n_faults_total"]}, '
         f'incl. dynamic switch-out faults, 2 seeds)',
         f'failing {r["n_fail"]} (window {r["n_fail_window"]}, outside {r["n_fail_outside"]}); first-order coefficient '
         f'c1 = {r["c1"]:.2f} (window {r["c1_window"]:.2f}, outside {r["c1_outside"]:.2f}); by block {r["by_block"]}')
    rows.append(dict(d=d, scenario='A' if pi else 'B', scope=scope, faults=r['n_tested'], failing=r['n_fail'],
                     c1_window=r['c1_window'], c1_outside=r['c1_outside']))
dyn = [f for f in N5[(3, 1.0, False)]['failures'] if f['fault'].startswith("('SO")]
print('failing dynamic switch-out faults at d=3 (A):', len(dyn))
print('%.0f s' % (time.time() - t0))
display(pd.DataFrame(rows).round(3))
pd.DataFrame(N5[(3, 1.0, False)]['failures'][:30])'''),
    md(r'''## N9. switch-out の物理スケジュール
switch-out の補正 $\mathrm{CZ}(\Gamma(a))$ は、全ての $S^g_v$ の CZ 対の和集合の辺彩色による**固定の層スケジュール**で適用する（層数は測定結果によらない最悪値で、時間の計数に使う）。(i) 各層で同じ量子ビットを 2 回使わない、(ii) どの $\kappa$ でも適用される CZ が $\prod_{v\in\kappa}S^g_v$ の CZ 部分（同じ対が偶数回なら相殺）と一致する、(iii) 時間の計数 `block_steps` が $g$ の読み出し 1 ＋層数であること、を確認する。'''),
    code(r'''for d in (3, 5, 7, 9):
    circ = D4SwitchingCircuit(d, 1, 1, 1)
    ok = True
    layers = {}
    for pr, c_ in circ.so_colour.items():
        layers.setdefault(c_, []).append(pr)
    for c_, prs in layers.items():
        qs = [q for pr in prs for q in pr]
        ok &= len(qs) == len(set(qs))
    r_ = np.random.default_rng(d)
    for _ in range(20):
        kap = r_.integers(0, 2, len(circ.gens_g))
        want = {}
        for v in np.nonzero(kap)[0]:
            for pr in circ.gpairs[v]:
                pr = tuple(sorted(pr)); want[pr] = want.get(pr, 0) ^ 1
        want = {pr for pr, c_ in want.items() if c_}
        ok &= want <= set(circ.so_colour)
    so = [b for b in circ.blocks if b['name'][0] == 'SO'][0]
    ok &= block_steps(circ, so) == 1 + len(layers)
    record(f'N9 d={d}: switch-out CZ schedule valid (no qubit used twice per layer, covers every kappa), '
           f'time = readout + {len(layers)} CZ layer(s)', ok, f'{len(circ.so_colour)} distinct CZ pairs')'''),
    md(r'''## N6. Litinski（2019）の誤り予算の再実装
公開コード（github.com/litinski/magicstates、Apache-2.0）は実行せずに読んで再実装した（`msd_noisy_lib`）。そのノートブックに保存されている出力（15-to-1 の 1 段 4 件、(15-to-1)×(15-to-1) の 2 段 4 件）を、出力誤り（有効数字 2 桁）・量子ビット数・code cycle 数（小数 1 桁）で再現することを確認する（128 ビット演算）。'''),
    code(r'''LIT_PUBLISHED = [  # (args, output error, qubits, code cycles) as printed in Python/magicstates.ipynb
    ((1e-4, 7, 3, 3), '4.4e-08', 810, '18.1'), ((1e-4, 9, 3, 3), '9.3e-10', 1146, '18.1'),
    ((1e-4, 11, 5, 5), '1.9e-11', 2066, '30.0'), ((1e-3, 17, 7, 7), '4.5e-08', 4618, '42.6'),
    ((1e-4, 9, 3, 3, 25, 9, 9, 4), '6.3e-25', 18630, '67.7'), ((1e-3, 11, 5, 5, 25, 11, 11, 6), '2.7e-12', 30732, '82.5'),
    ((1e-3, 13, 5, 5, 29, 11, 13, 6), '3.3e-14', 39108, '97.5'), ((1e-3, 17, 7, 7, 41, 17, 17, 6), '4.5e-20', 73460, '127.5')]
rows = []
for args, e_pub, q_pub, c_pub in LIT_PUBLISHED:
    f = litinski2019_one_level if len(args) == 4 else litinski2019_two_level
    r = f(*args, backend='mp')
    ok = ('%.1e' % r['eps'] == e_pub) and r['qubits'] == q_pub and ('%.1f' % r['cycles'] == c_pub)
    rows.append(dict(protocol=r['protocol'], p=args[0], eps=r['eps'], eps_published=e_pub, qubits=r['qubits'],
                     qubits_published=q_pub, cycles=round(r['cycles'], 2), cycles_published=c_pub, match=ok))
    record(f'N6 Litinski 2019 {r["protocol"]} at p={args[0]:g} reproduces the published output', ok,
           f'eps {r["eps"]:.2e} (published {e_pub}), qubits {r["qubits"]} ({q_pub}), cycles {r["cycles"]:.2f} ({c_pub})')
pd.DataFrame(rows)'''),
    md(r'''## N7. 表面符号の較正回路
`noisy_surface_code` が stim の回転表面符号メモリ回路を D4 と同じ雑音モデルに置き換えたものであることを確認する：雑音 0 で全ての検出器と観測量が 0（決定的）、回路レベルの最短の論理誤り（graph-like）が $d$、1 ラウンドが 6 時間ステップ。'''),
    code(r'''for d in (3, 5, 7):
    ok, det = True, []
    for basis in ('x', 'z'):
        c0 = noisy_surface_code(d, d, basis, 0.0)
        s = c0.compile_detector_sampler(seed=1).sample(200, append_observables=True)
        c1 = noisy_surface_code(d, d, basis, 1e-3)
        dist = len(c1.detector_error_model(decompose_errors=True).shortest_graphlike_error())
        c2 = noisy_surface_code(d, 2 * d, basis, 1e-3)
        steps = (c2.num_ticks - c1.num_ticks) / d
        ok &= (not s.any()) and dist == d and steps == 6
        det.append(f'{basis}: distance {dist}, {steps:g} steps/round')
    record(f'N7 d={d}: calibration circuit deterministic without noise, circuit distance = d, 6 time steps per round', ok, '; '.join(det))'''),
    md(r'''## N8. GoSC 15-to-1 の誤り予算モデル
(i) 雑音 0 で出力誤り 0・成功確率 1、(ii) 表面符号の誤りを 0 にして入力の魔法状態に確率 $p$ の $Z$ 誤りだけを入れると、出力誤りが 15-to-1 の既知の値 $35p^3$（主要項）になる — Fig. 15 の 11 個の回転（`GOSC_ROTATIONS`、第 2 部 M1 で照合済みの行列から）が正しい符号を構成していることの確認。'''),
    code(r'''zero = lambda p, d: mpmath.mpf(0)
ps0, e0 = run_steps(gosc_15to1_steps(1e-3, 7, zero, p_in=0), out_qubit=4, backend='mp')
record('N8 GoSC 15-to-1 without errors: output error 0, success probability 1', abs(e0) < 1e-30 and abs(ps0 - 1) < 1e-30)
for p in (1e-3, 1e-4):
    ps1, e1 = run_steps(gosc_15to1_steps(p, 7, zero, p_in=p, in_channel='z'), out_qubit=4, backend='mp')
    ratio = float(e1 / (35 * mpmath.mpf(p) ** 3))
    record(f'N8 GoSC 15-to-1 with Z errors (prob. {p:g}) on the input states only: output error = 35 p^3 (leading order)',
           abs(ratio - 1) < 0.02, f'ratio {ratio:.4f}, success probability {float(ps1):.6f}')'''),
    md(r'''## まとめ'''),
    code(r'''n_pass = sum(v['status'] == 'PASS' for v in VAL.values()); n_fail = sum(v['status'] == 'FAIL' for v in VAL.values())
print(f'{n_pass} PASS, {n_fail} FAIL, {len(VAL) - n_pass - n_fail} INFO')
json.dump(dict(notebook='validation', results=VAL,
               not_verified=['Knill-Laflamme distance of the D4 phase for L >= 6 (L >= 7 with FULL_KL=True)',
                             'exact (state-vector) simulation of the noisy protocol for L >= 3 (N3 is L = 1, 2; for L >= 3 the Pauli-proxy method is justified by the conditions of N1 and Takada et al.)',
                             'N3b replaces S0 by its ideal output (S0 is Clifford and exact in stim, but its exact state vector does not fit in memory)',
                             'N1 is a finite-size algebraic check (L = 2..7, 9), not a proof for general L',
                             'fault tolerance: single faults are NOT always corrected by our circuit + decoders (N5); only the sampled measurement branches are tested',
                             'causes of the single-fault failures not fully classified (decoder vs gadget vs ordering vs boundary)',
                             'feed-forward X corrections are applied as a noiseless, zero-time Pauli layer (idealisation)',
                             'constants of the O(d) round counts (kappa)',
                             'FSC -> standard rotated-patch conversion',
                             'stabilizer-level layout of the Z_m (x) Y_0 twist-based seam (O(d) measurement qubits not counted)',
                             'error budget of the GoSC layouts (our adaptation of Litinski 2019, not simulated at circuit level)',
                             'classical processing latency']),
          open(f'{RESULTS}/validation.json', 'w'), indent=1)
pd.DataFrame([(k, v['status']) for k, v in VAL.items()], columns=['check', 'status'])'''),
    md(r'''### 検証できていない事項（正しさが確認されたものとして扱わない）
* $L\ge 6$ の D4 相の Knill–Laflamme 距離（$L\le5$ は V5b で厳密に $=L$）。
* $L\ge3$ での雑音付きプロトコルの状態ベクトルによる直接照合（N3 は $L\le2$。$L\ge3$ の Pauli proxy 法の正しさは N1 の有限サイズの条件確認と Takada らの定理に依拠）。N3b は S0 を理想出力で置き換えている。
* 耐故障性：本研究の回路・復号器は 1 個の故障を必ずしも訂正しない（N5）。N5 は 2 つの測定分岐しか試していない。失敗の原因（復号器・測定ガジェット・順序・境界）の完全な分類はしていない。
* フィードフォワードの $X$ 補正は雑音なし・時間 0 の Pauli 層として適用する理想化。
* $O(d)$ ラウンドの定数 $\kappa$、FSC → 通常の回転表面符号パッチへの変換。
* Fig. 17c の継ぎ目（$Z_m\otimes Y_0$ の twist）の安定化子レベルの配置、GoSC 配置の誤り予算（Litinski 2019 の方法を本研究で GoSC の形に当てはめたもの。回路レベルのシミュレーションではない）。
* 古典処理の待ち時間。
'''),
]
save('validation.ipynb', cells)
