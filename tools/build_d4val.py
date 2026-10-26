from nbutil import *

cells = [md(r'''
# D4_switching_validation — Kobayashi方式の正当性検証

メインの資源計数（`D4_switching_main.ipynb`）とは**独立に**、符号・論理ゲート・プロトコル・測定回路の正しさを確認する。
各検査は `record(name, passed, detail)` で記録し、最後に `results/validation_d4.json` に保存する（比較ノートはこれを読み、検証状況を表示する）。

| ID | 内容 | 方法 |
|---|---|---|
| V1 | 安定化子の定義が [KZH] と一致 | Fig. 4 から読み取った特徴（境界ごとの生成元の種類、CZ の位置）との照合 |
| V2 | 符号空間の整合性 | $C_Z$ 保存・$C_Z$ 上の可換性・$S^2=1$・$k=1$（$\mathbb F_2$ 厳密）＋負の対照 |
| V3 | 独立実装による照合 | $L=1$ を 埋め込みの D4 ライブラリを使わず密ベクトルで実装し、符号次元と論理 $T^\dagger$ を確認 |
| V4 | 論理演算子と距離 | MILP の距離を全探索で照合（小さい $L$） |
| V5 | Knill–Laflamme 条件 | $L=2$ の全 weight≤2 Pauli |
| V5b | D4 相の KL 距離 $=L$ | $L=3..5$（6）：低重み安定化子シフトの厳密列挙＋対角分解による厳密下界 |
| V6 | transversal $U$ | $USU^\dagger=S'$ の局所厳密列挙、論理位相、$V(C_Z)=C_Z$、規約を変えた負の対照 |
| V7 | プロトコル全体（厳密） | $L=1,2$ 疎状態ベクトル、ランダム入力・ランダム分岐、補正を外した負の対照、測定結果の統計 |
| V7b | 回路と位相公式の照合 | $L=3,4$：実際の測定・$U$・補正回路を配置入力に作用させ、同じ分岐での振幅比が $F$ と一致 |
| V8 | プロトコル全体（大きな $L$） | 配置ごとの位相 $F(a)$ が論理剰余類上で一定であることの局所厳密証明（$L\le 8$） |
| V9 | 測定ガジェット | 制御-$S$ 回路＋$X$ 測定 = 理想射影測定（密行列） |
| V10 | スケジュールの妥当性 | 同一クラス内の異なるガジェットのゲートが可換（小行列で独立に確認）、$U$ のゲートが互いに可換 |
| V11 | 資源計数の独立照合 | 幾何からの直接公式との一致 |
'''),
    code(SETUP), code(INSTALL),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py'),
    code(r'''import time, itertools
VAL = {}
def record(name, passed, detail=''):
    detail = str(detail).replace('\n', ' ')
    VAL[name] = dict(status='PASS' if passed else 'FAIL', detail=detail)
    print(('PASS ' if passed else 'FAIL ') + name + (f'  | {detail}' if detail else ''))
QUICK = False
L_MAX_INVARIANCE = 5 if QUICK else 8
rng = np.random.default_rng(SEED)'''),
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
    lb, ts, det = kl_distance_lower_bound(P_.d4, L - 1, P_.frozen)
    dz, dx, _, _ = distances(P_.d4)
    ok = (lb is None or lb >= L) and dz == L and dx >= L
    record(f'V5b L={L}: every Pauli error of weight <= L-1 satisfies Knill-Laflamme -> D4-phase distance = L',
           ok, f'{len(ts)} stabilizer shifts of weight <= {L-1} enumerated; min LB_t = {lb}; d_Z = {dz}; X-shift = {dx}; {time.time()-t0:.0f}s')"""),
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
    md(r'''## まとめ'''),
    code(r'''n_pass = sum(v['status'] == 'PASS' for v in VAL.values()); n_fail = len(VAL) - n_pass
print(f'{n_pass} PASS, {n_fail} FAIL')
json.dump(dict(notebook='D4_switching_validation', results=VAL,
               not_verified=['Knill-Laflamme distance of the D4 phase for L >= 6 (L >= 7 with FULL_KL=True)',
                             'full code-state simulation of the protocol for L >= 3 (V7b checks the circuit on configuration inputs, V8 proves the phase formula locally)',
                             'fault tolerance / JIT decoder performance (noise not simulated)',
                             'constants of the O(d) round counts',
                             'FSC -> standard rotated-patch conversion']),
          open(f'{RESULTS}/validation_d4.json', 'w'), indent=1)
pd.DataFrame([(k, v['status']) for k, v in VAL.items()], columns=['check', 'status'])'''),
    md(r'''### 検証できていない事項（正しさが確認されたものとして扱わない）
* $L\ge 6$ の D4 相の Knill–Laflamme 距離（$L\le5$ は V5b で厳密に $=L$；`FULL_KL=True` で $L=6$ も確認可能、約7分）。
* $L\ge3$ での符号状態全体の厳密シミュレーション（代わりに V7b：回路と位相公式の照合、V8：位相公式の局所厳密証明）。
* 耐故障性（JIT decoder の性能、しきい値）：ノイズを入れていないため未検証。
* $O(d)$ ラウンドの定数。
* FSC → 通常の回転表面符号パッチへの変換。
'''),
]
save('D4_switching_validation.ipynb', cells)
