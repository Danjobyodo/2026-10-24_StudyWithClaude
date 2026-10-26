from nbutil import *

cells = [md(r'''
# D4_switching_main — Kobayashi方式：D4符号へのcode switchingによる $|T_L\rangle$ 生成の資源計数

**目的**：Kobayashi–Zhu–Hsin [KZH] のプロトコル（folded surface code (FSC) → D4符号 → transversal $U$（論理 $T^\dagger$）→ FSC）を、境界付き正方格子上で**ゲートレベルの回路として構成**し、段階ごとの物理量子ビット数・ラウンド数・物理時間・時空間体積・ゲート内訳を数える。

**このノートでは正しさの検証はしない**（→ `D4_switching_validation.ipynb`）。ここで数えた回路が正しい論理操作を実装していることは、検証ノートで別途確認する。

**理想条件**：物理ノイズ0。測定結果のランダム性は残る（$S^g_X$ の結果 $m_v$、$Z^g$ の結果）。

### プロトコル（[KZH] 本文 p.4 の手順 1–4、SM Sec. V A）と本研究での段階分け
| 段階 | 内容 | 根拠 | 時間 |
|---|---|---|---|
| S0 | FSC の $\overline{|+\rangle}$ 準備（$r,b$ を $|+\rangle$、$g$ を $|0\rangle$）。ノイズなしでは Z-plaquette のランダムな結果を $X$ で補正 | [KZH] 手順1（準備方法は記述なし） | `FT_ref`: $\lceil\kappa_0 d\rceil$ ラウンド（**仮定**）、`NL_min`: 1 |
| S1 | 全D4安定化子（$S^g_X$ を含む）の測定 = gauging | [KZH] 手順2「$O(d)$ rounds」 | `FT_ref`: $\lceil\kappa_{in} d\rceil$（**仮定**）、`NL_min`: $S^g_X$ のみ1ラウンド |
| S2 | $U = WV$（式(5)(8)） | [KZH] 手順3 | 定数深さ |
| S3 | $Z^g_e$ 測定 → $\prod_{v\in\kappa}S^g_{X;v}$ の CZ 部分で補正 | [KZH] 手順4 | 定数深さ |
| S4 | FSC のEC | [KZH] 手順4「$O(d)$ rounds」 | `FT_ref`: $\lceil\kappa_{out} d\rceil$（**仮定**）、`NL_min`: 0 |
| S5 | 論理 $\bar X$（Pauliフレーム）で $T^\dagger|+\rangle\to T|+\rangle$ | 本研究 | 0 |
| S6 | FSC → 通常の回転表面符号パッチ（比較Bのみ） | **未解決** | 除外（下界） |

ノイズなしで必要な追加処理（検証ノートで正しさを確認）：$m_v=1$ に対する $Z^rZ^b$ 文字列による補正は Pauli フレームで行う（物理コスト0）。

**距離**：$L=d$（FSC の $Z$ 距離が $L$、$X$型シフト距離が $L+3$ であることを下で MILP により確認）。
'''),
    code(SETUP), code(INSTALL), code(PLOTSTYLE),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py'),
    md(r'''## パラメータ
* `D_LIST`：評価する符号距離（$L=d$）。
* `DIST_L_MAX`：厳密MILPで符号距離を求める最大の $L$（$L=9,10$ は1点あたり数十秒〜2分）。
* `DUR`：各原始ゲートの所要時間（既定は全て1単位＝**仮定**）。
* `PARAMS`：$\kappa_0,\kappa_{in},\kappa_{out}$（$O(d)$ の定数；**文献で未確定のため仮定**）など。
* `QUICK=True` で小規模動作確認。'''),
    code(r'''QUICK = False
D_LIST = [3, 5, 7, 9, 11, 13, 15]
DIST_L_MAX = 8
if QUICK:
    D_LIST, DIST_L_MAX = [3, 5, 7], 4
DUR = dict(DUR_UNIT)              # e.g. DUR['CCZ'] = 3 to explore slower 3-qubit gates
PARAMS = dict(DEFAULT_PARAMS)
print(PARAMS)'''),
    md(r'''## 1. 格子・境界・安定化子の構成（[KZH] Fig. 4 の再現）

境界頂点で許される $X$ 型生成元は、「$X$ 部分が境界の $Z$ 条件（$\mathcal L_r$: $Z^r$、$\mathcal L_b$: $Z^b$、$\mathcal L_{rb}$: $Z^rZ^b, Z^g$）と可換になる色の組」として**自動導出**している（手で境界生成元を書き下していない）。結果は Fig. 4 と一致する：
$\mathcal L_r$ 上は $S^g, S^b$、$\mathcal L_b$ 上は $S^r, S^g$、左上角は $S^g$、$\mathcal L_{rb}$ 上は $S^rS^b$、左下角・右上角は無し。'''),
    code(r'''L_FIG = 9
lat = D4Lattice(L_FIG)
types = {}
for v in lat.vertices:
    types[v] = '+'.join(''.join(CNAME[c] for c in combo) for combo in lat.vertex_x_types(v)) or '—'
from collections import Counter as _C
print('X-generator types per vertex class (L=9):')
cls = {}
for v, t in types.items():
    x, y = v
    key = ('bulk' if 0 < x < L_FIG and 0 < y < L_FIG else
           'corner(0,0)' if v == (0, 0) else 'corner(0,L)' if v == (0, L_FIG) else
           'corner(L,L)' if v == (L_FIG, L_FIG) else 'corner(L,0)' if v == (L_FIG, 0) else
           'L_r (left)' if x == 0 else 'L_b (top)' if y == L_FIG else 'L_rb (bottom/right)')
    cls.setdefault(key, set()).add(t)
for k, v in cls.items():
    print(f'  {k:22s}: {sorted(v)}')

fig = plt.figure(figsize=(14, 4.6))
gsp = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
ax = fig.add_subplot(gsp[0]); ax.grid(False)
bcol = {'Lr': C_R, 'Lb': C_B, 'Lrb': '#6250d6', None: '#b9b8b2'}
for e in range(lat.E):
    (x0, y0), (x1, y1) = lat.endpoints(e)
    bd = lat.boundary_of_edge(e)
    ax.plot([x0, x1], [y0, y1], color=bcol[bd], lw=3 if bd else 0.8, zorder=1)
mk = {'r+g+b': ('o', '#52514e'), 'g+b': ('s', C_G), 'r+g': ('D', C_G), 'g': ('^', C_G), 'rb': ('P', '#6250d6'), '—': ('x', '#898781')}
seen = set()
for v, t in types.items():
    m, c = mk[t]
    ax.scatter(*v, marker=m, color=c, s=30, zorder=3, label=None if t in seen else f'X gens: {t}')
    seen.add(t)
ax.set_title('Boundaries & allowed X-type generators (L=9)')
ax.text(-0.9, L_FIG / 2, r'$\mathcal{L}_r$', color=C_R, fontsize=13, ha='center')
ax.text(L_FIG / 2, L_FIG + 0.6, r'$\mathcal{L}_b$', color=C_B, fontsize=13, ha='center')
ax.text(L_FIG / 2, -1.0, r'$\mathcal{L}_{rb}$', color='#6250d6', fontsize=13, ha='center')
ax.set_aspect('equal'); ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.08), fontsize=7, ncol=3)
ax.set_xlim(-1.5, L_FIG + 0.5); ax.set_ylim(-1.5, L_FIG + 1.2); ax.set_xticks([]); ax.set_yticks([])
lat4 = D4Lattice(3)
def mid(q):
    (x0, y0), (x1, y1) = lat4.endpoints(q // 3)
    return (x0 + x1) / 2, (y0 + y1) / 2
for k, (c, col) in enumerate([(R, C_R), (G, C_G), (B, C_B)]):
    ax = fig.add_subplot(gsp[k + 1]); ax.grid(False)
    for e in range(lat4.E):
        (x0, y0), (x1, y1) = lat4.endpoints(e)
        ax.plot([x0, x1], [y0, y1], color='#b9b8b2', lw=0.8)
    v = (1, 1) if c != B else (2, 2)
    if c == G: v = (1, 1)
    op = lat4.SX(v, c)
    ax.scatter(*v, color='#0b0b0b', s=25, zorder=3)
    for qq in op.s:
        x, y = mid(qq); ax.text(x, y, 'X', color=col, ha='center', va='center', fontsize=12, weight='bold')
    for (i, j) in op.quad:
        (xa, ya), (xb, yb) = mid(i), mid(j)
        ax.plot([xa, xb], [ya, yb], ls='--', color='#52514e', lw=1.3)
        cols_ = ''.join(sorted([CNAME[i % 3], CNAME[j % 3]], key='rgb'.index))
        ax.text((xa + xb) / 2 + 0.05, (ya + yb) / 2 - 0.12, f'CZ$^{{{cols_[0]},{cols_[1]}}}$', fontsize=8, color='#52514e')
    ax.set_title(f'$S^{CNAME[c]}_{{X;v}}$, v={v}')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_lattice_and_stabilizers.png', dpi=150); plt.show()'''),
    md(r'''右図：$S^r_X$ の CZ$^{g,b}$ は頂点の右上の plaquette（$v$ = 頂点0）、$S^g_X$ の CZ$^{r,b}$ は左上（$v$ = 頂点1）と右下（$v$ = 頂点2）、$S^b_X$ の CZ$^{r,g}$ は左下（$v$ = 頂点3）。これは [KZH] Fig. 4 の破線の位置と一致する。

## 2. 符号データ：生成元数・論理量子ビット数・符号距離

* $k$：$\dim C_Z - \mathrm{rank}(\text{shifts})$（生成元の $C_Z$ 上での可換性・$S^2=1$ と併せて検証ノートで確認）。
* $d_Z$：論理 $Z$ 型 Pauli の最小重み。$d_X^{\rm shift}$：$C_Z$ に属し安定化子シフトの張る空間に属さない $X$ シフトの最小重み（D4相では CZ 修飾があるため、これは Knill–Laflamme 距離そのものではない。$L=2$ のみ KL 条件を直接検査）。
* いずれも厳密 MILP（HiGHS）。'''),
    code(r'''import time
rows = []
for L in range(1, DIST_L_MAX + 1):
    t0 = time.time()
    lat = D4Lattice(L)
    d4 = CodeData(lat); fsc = CodeData(lat, (R, B), dressed=False)
    c4, cf = code_consistency(d4), code_consistency(fsc)
    dz4, dx4, _, _ = distances(d4); dzf, dxf, _, _ = distances(fsc)
    rows.append(dict(L=L, D4_qubits_incl_frozen=d4.lat.n, D4_xgens=len(d4.xgens), D4_k=c4['k'],
                     D4_commuting_on_CZ=not c4['noncommuting_pairs'], D4_dZ=dz4, D4_dX_shift=dx4,
                     FSC_k=cf['k'], FSC_dZ=dzf, FSC_dX=dxf, seconds=round(time.time() - t0, 2)))
code_df = pd.DataFrame(rows)
code_df.to_csv(f'{RESULTS}/d4_code_table.csv', index=False)
code_df'''),
    md(r'''**結果**：調べた全ての $L$ で $k=1$、$d_Z = L$、$X$ 型シフト距離 $= L+3$（D4・FSC とも）。
* FSC（CSS 符号）の符号距離は厳密に $\min(d_Z,d_X)=L$。
* D4 相（非 Pauli 符号）では $d_Z=L$ の論理 $Z$ があるので距離 $\le L$。下界は検証ノート V5b で、重み $\le L-1$ の全 Pauli が Knill–Laflamme 条件を満たすことを $L\le5$（`FULL_KL=True` で $L=6$）について厳密に示した（$=L$）。それより大きな $L$ では D4 相の距離は**未証明**（上界 $L$ のみ）。
* よって比較の横軸は **$L$（= FSC の距離 = 入出力符号の距離）** であり、「D4 相も距離 $L$」は $L\le5$（6）でのみ証明済みである。$X$ 方向には $3$ だけ余分な距離があり、正方形でない格子で量子ビット数を削れる可能性があるが、[KZH] の Fig. 4 の正方形配置に従う（最適化はしていない）。

## 3. 段階別の資源計数

`D4CostData(L)` は明示的な格子上で、各段階の回路を**ゲート単位で構成**して数える：
* ラウンド回路：各生成元に測定補助1個。$X$ 型 Clifford 安定化子 $S = X(s)\prod CZ$ は「補助 $|+\rangle$ → CNOT（補助→データ） → CCZ（補助, $i$, $j$） → $X$ 測定」（制御-$S$ による標準的な測定。**この回路は [KZH] には書かれておらず本研究の構成**。正しさは検証ノートで密行列により確認）。
* frozen 量子ビット（境界の単一量子ビット $Z$ 安定化子で $|0\rangle$ に固定）は割り当てず、それに作用するゲート（恒等になる）も除く。$\mathcal L_{rb}$ の $g$ 量子ビットを除くため、そこでの $V$ の CNOT 対（$Z^rZ^b=+1$ 上で恒等）も除く。
* S3 の CZ 補正は最悪ケース（全頂点）で数える（期待値はその半分；$\kappa$ が一様であることは検証ノートで統計的に確認）。'''),
    code(r'''stage_df, sum_df, blocks = d4_resource_tables(D_LIST, DUR, PARAMS)
stage_df.to_csv(f'{RESULTS}/d4_stage_costs.csv', index=False)
sum_df.to_csv(f'{RESULTS}/d4_summary.csv', index=False)
json.dump(dict(params=PARAMS, dur=DUR, blocks=blocks), open(f'{RESULTS}/d4_blocks.json', 'w'), indent=1)
print('saved', len(stage_df), 'stage rows')'''),
    md(r'''### 3.1 段階別の表（$d=9$、`FT_ref`、比較A、原始ゲートセット、データ＋測定補助）
`rounds` は符号ごとのラウンド数（S0, S4 は FSC のラウンド、S1 は D4 のラウンド）。`time_lb/ub` は物理時間（全ゲート1単位の仮定）の下界／上界。`source` は値の出所（文献／明示的回路からの計数／仮定パラメータ／未解決）。'''),
    code(r'''d_show = 9 if 9 in D_LIST else D_LIST[-1]
view = stage_df[(stage_df.d == d_show) & (stage_df.model == 'FT_ref') & (stage_df.comparison == 'A') &
                (stage_df.gateset == 'primitive') & (~stage_df.data_only)]
view[['stage', 'qubits', 'rounds', 'time_lb', 'time_ub', 'gates', 'source']]'''),
    md(r'''### 3.2 1ラウンドあたりのコストと定数深さ段階'''),
    code(r'''rows = []
for d in D_LIST:
    b = blocks[f'{d}|primitive']; bd = blocks[f'{d}|clifford_t']
    rows.append(dict(d=d, FSC_round_lb=b['round']['FSC']['time_lb'], FSC_round_ub=b['round']['FSC']['time_ub'],
                     D4_round_lb=b['round']['D4']['time_lb'], D4_round_ub=b['round']['D4']['time_ub'],
                     D4_round_ub_cliffordT=bd['round']['D4']['time_ub'], D4g_round_ub=b['round']['D4g']['time_ub'],
                     CCZ_per_D4_round=b['round']['D4']['gates'].get('CCZ', 0),
                     CNOT_per_D4_round=b['round']['D4']['gates'].get('CNOT', 0),
                     U_depth_time=b['U_time'], switch_out_time=b['out_time'],
                     U_CS=b['U_gates'].get('CS', 0) + b['U_gates'].get('CSDG', 0), U_TDG=b['U_gates'].get('TDG', 0)))
per_round_df = pd.DataFrame(rows); per_round_df'''),
    md(r'''## 4. 符号距離に対するスケーリング（厳密な多項式の同定）

数値はすべて明示的な格子上の計数なので、$L$ の多項式になることが期待される。最初の4点で補間した（高々3次の）多項式が**残りすべての点で厳密に一致するか**を確認する（一致しなければ「多項式ではない／境界の効果がまだある」と表示）。これは評価範囲の $L$ についての恒等式の確認であり、全ての $L$ についての証明ではない（ただし量子ビット数・ゲート数は格子の局所構造から $L$ の2次式、体積は $\lceil\kappa d\rceil$ ラウンドを掛けて3次式になる）。'''),
    code(r'''import sympy
x = sympy.Symbol('d')
def identify(values):
    Ls = sorted(values)
    if len(Ls) < 5:
        return None, False
    p = sympy.interpolate([(L, values[L]) for L in Ls[:4]], x)      # cubic through 4 points
    ok = all(sympy.nsimplify(p.subs(x, L) - values[L]) == 0 for L in Ls)
    return sympy.expand(p), ok
quant = {
    'peak qubits (D4 phase: data+anc)': {d: blocks[f'{d}|primitive']['n_total'] for d in D_LIST},
    'FSC qubits (data+anc)': {d: blocks[f'{d}|primitive']['n']['fsc_data'] + blocks[f'{d}|primitive']['n']['fsc_anc'] for d in D_LIST},
    'D4 data qubits': {d: blocks[f'{d}|primitive']['n']['d4_data'] for d in D_LIST},
    'CCZ per D4 round': {d: blocks[f'{d}|primitive']['round']['D4']['gates'].get('CCZ', 0) for d in D_LIST},
    'CNOT per D4 round': {d: blocks[f'{d}|primitive']['round']['D4']['gates'].get('CNOT', 0) for d in D_LIST},
}
for comp in ['A']:
    for model in ['FT_ref', 'NL_min']:
        sel = sum_df[(sum_df.model == model) & (sum_df.comparison == comp) & (sum_df.gateset == 'primitive') & (~sum_df.data_only)]
        quant[f'V [qubit*round] {model}'] = dict(zip(sel.d, sel.V_qubit_round))
        quant[f'V [qubit*time] lb {model}'] = dict(zip(sel.d, sel.V_qubit_time_lb))
        quant[f'V [qubit*time] ub {model}'] = dict(zip(sel.d, sel.V_qubit_time_ub))
        quant[f'rounds {model}'] = dict(zip(sel.d, sel.rounds))
poly_rows = []
for k, v in quant.items():
    p, ok = identify({d: int(round(val)) for d, val in v.items()})
    poly_rows.append(dict(quantity=k, polynomial_in_d=str(p), exact_on_all_points=ok))
poly_df = pd.DataFrame(poly_rows); poly_df.to_csv(f'{RESULTS}/d4_polynomials.csv', index=False); poly_df'''),
    md(r'''`FT_ref` の体積の多項式は $\kappa_0=\kappa_{in}=\kappa_{out}=1$（**仮定**）に対するもの。一般の $\kappa$ では、主要項は
$$V_{\rm qubit\cdot round}\simeq \big(8\kappa_0 + 12\kappa_{in} + 8\kappa_{out}\big)\,d^3$$
（FSC 相の量子ビット数 $=8d^2+4d-1$、D4 相 $=12d^2+4d-1$、いずれもデータ＋測定補助）。

## 5. 図：段階別の内訳と量子ビット数'''),
    code(r'''sel = stage_df[(stage_df.d == d_show) & (stage_df.model == 'FT_ref') & (stage_df.comparison == 'A') &
               (stage_df.gateset == 'primitive') & (~stage_df.data_only)].copy()
sel['V_round'] = sel.qubits * sel.rounds
sel['V_time_lb'] = sel.qubits * sel.time_lb
sel['V_time_ub'] = sel.qubits * sel.time_ub
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
lab = [s.split(' ')[0] for s in sel.stage]
axes[0].barh(lab, sel.V_round, color=C_KOB, height=0.6)
axes[0].invert_yaxis(); axes[0].set_xlabel('qubit · round  (κ=1 assumed)')
axes[0].set_title(f'Space-time volume per stage, d={d_show} (FT_ref, A)')
for i, v in enumerate(sel.V_round):
    axes[0].text(v, i, f' {int(v):,}', va='center', fontsize=8, color='#52514e')
axes[1].barh(lab, sel.V_time_ub, color=C_KOB, alpha=0.35, height=0.6, label='upper bound (valid schedule)')
axes[1].barh(lab, sel.V_time_lb, color=C_KOB, height=0.6, label='lower bound')
axes[1].invert_yaxis(); axes[1].set_xlabel('qubit · time-unit (all gates = 1 unit, assumed)')
axes[1].set_title('Physical-time volume per stage'); axes[1].legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_stage_breakdown.png', dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ds = np.array(D_LIST)
ax.plot(ds, [blocks[f'{d}|primitive']['n_total'] for d in D_LIST], 'o-', color=C_KOB, label='D4 phase (peak)')
ax.plot(ds, [blocks[f'{d}|primitive']['n']['fsc_data'] + blocks[f'{d}|primitive']['n']['fsc_anc'] for d in D_LIST],
        's--', color=C_AUX, label='FSC phase')
ax.set_xlabel('L  (= FSC code distance d)'); ax.set_ylabel('physical qubits (data + ancilla)')
ax.set_title('Kobayashi protocol: qubits per phase (counted)'); ax.legend()
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_qubits_vs_d.png', dpi=150); plt.show()'''),
    md(r'''## 6. 解釈と注意

* **数えたもの**：明示的な格子・回路から数えた量子ビット数、ゲート数、ラウンドあたりの物理時間（下界・上界）。
* **仮定に依存するもの**：`FT_ref` のラウンド数（$\kappa$）。[KZH] は $O(d)$ としか述べておらず、JIT decoder の文献 [Dav] の全文は本環境から参照できなかった。ゲート所要時間（全て1単位）。
* **数えていないもの**：JIT decoder／matching・RG decoder の古典計算時間、制御系の遅延、FSC→通常パッチ変換（比較Bで未解決として扱う）。
* D4 相の 1 ラウンドは CCZ を含むため、FSC や回転表面符号の 1 ラウンドより深い（上の表）。**異なる符号のラウンド数を同一の時間単位とみなさない**ため、比較では物理時間でも示す。
* `FT_ref` では物理 CCZ が $6(d-1)^2$ 個／ラウンド × $\lceil\kappa_{in}d\rceil$ ラウンド $=\Theta(d^3)$ 必要になる。ノイズなしでは問題にならないが、ノイズありの拡張では重要な資源になる（比較Bで T 数としても表示）。
'''),
]
save('D4_switching_main.ipynb', cells)
