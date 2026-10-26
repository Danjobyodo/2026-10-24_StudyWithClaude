from nbutil import *

val = [md(r'''
# MSD_15to1_validation — Litinski 15-to-1 の正当性・計数の検証

| ID | 内容 |
|---|---|
| M1 | 式(2) の三直交性、式(4) が式(2) と（行変形＋列の並べ替えで）同値であること、式(4)→式(1)（puncturing）、式(1) → Fig. 15 の 11 回転（図からの手転記と照合） |
| M2 | 理想入力 → 測定結果は決定的に +1、出力 $T^\dagger|+\rangle$（[Lit] "actually $T^\dagger$"） |
| M3 | 15 量子ビット符号（Fig. 14）：transversal $T$ = 論理 $T^\dagger$、5 量子ビット回路と同じ出力 |
| M4 | 誤り検出：1・2個の $Z$ 誤りは全て検出、3個で未検出 35 通り（$35p^3$）、未検出の 35 通りは全て誤った出力 |
| M5 | selective π/4（Fig. 17a）と auto-corrected π/8（Fig. 17b）：選んだ 4 種の Pauli 積とランダム入力について、全測定分岐で Pauli 補正のみで正しい |
| M6 | tile 数 11、time step 11、$121d^3$ の再計数（2通りの数え方） |
| M7 | 物理レベルの格子手術（stim）：2 パッチ＋1 tile の合併で $\bar Z_1\bar Z_2$ が測定される；量子ビット増分 |
| M8 | 回転表面符号パッチの量子ビット数・CNOT 数の閉形式、tile↔物理の換算の一貫性 |
| M9 | **Fig. 17c の 11 個の多パッチ測定 $P\otimes Z_m$ の物理レベル構成**：$\mathbb F_2$ による厳密判定（$d=3,5$ は合併符号の距離も MILP、$d=7$）、stim による測定値の照合（$d=3,5$）、高速版との一致 |
'''),
    code(SETUP), code(INSTALL),
    *embedded_lib_cells('common_lib.py', 'msd_lib.py'),
    code(r'''import itertools
VAL = {}
def record(name, passed, detail=''):
    detail = str(detail).replace('\n', ' ')
    VAL[name] = dict(status='PASS' if passed else 'FAIL', detail=detail)
    print(('PASS ' if passed else 'FAIL ') + name + (f'  | {detail}' if detail else ''))
rng = np.random.default_rng(SEED)'''),
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
    code(r'''n_pass = sum(v['status'] == 'PASS' for v in VAL.values())
print(f'{n_pass} PASS, {len(VAL) - n_pass} FAIL')
json.dump(dict(notebook='MSD_15to1_validation', results=VAL,
               not_verified=['stabilizer-level layout of the Z_m (x) Y_0 twist-based seam (O(d) measurement qubits not counted)',
                             'merged-code distance verified only for d = 3, 5 (validity criteria also for d = 7)',
                             'classical processing latency', 'Fig. 18 variant (extra 2x2 blocks) not modelled']),
          open(f'{RESULTS}/validation_msd.json', 'w'), indent=1)
pd.DataFrame([(k, v['status']) for k, v in VAL.items()], columns=['check', 'status'])'''),
]
save('MSD_15to1_validation.ipynb', val)
