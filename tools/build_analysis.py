"""Builds resource_analysis.ipynb: Part I (Kobayashi D4 switching), Part II (Litinski 15-to-1), Part III (comparison)."""
from nbutil import *

cells = [
    md(r'''
# resource_analysis — Kobayashi方式（D4 code switching）と Litinski方式（15-to-1）の資源解析・比較

論理魔法状態 $|T_L\rangle$ を 1 個作る 2 つのプロトコルを、**同じ計数ルール**でゲートレベルまで構成し、物理量子ビット数・時間・時空間体積を数えて比較する。物理ノイズ 0 の理想条件。

**このノートは単独で実行できる**（共通コードを下に埋め込んでおり、他のファイルを読み込まない）。正しさの検証は独立した 2 本の validation ノート（`D4_switching_validation.ipynb`, `MSD_15to1_validation.ipynb`）で行う。このノートは検証結果ファイル（`results/validation_*.json`）が同じフォルダにあれば読み込んで表示する。

| 部 | 内容 |
|---|---|
| §0 | パラメータ、両方式の資源表の計算（1 回だけ）、検証状況 |
| 第I部 | Kobayashi 方式の詳細：格子・安定化子（[KZH] Fig. 4 の再現）、符号距離、段階別資源、1 ラウンドのコスト、多項式 |
| 第II部 | Litinski 方式の詳細：論理回路、Fig. 17c の tile 配置、tile→物理の対応、資源、多項式、占有の時系列 |
| 第III部 | 比較：段階別一覧、主比較、スケーリング、$\kappa$ の損益分岐、CCZ 時間の感度、`NL_min`、比較B、仮定一覧、結論 |

### 結果の位置づけ（最初に読むこと）
数値は、**特定の仮定（とくに $O(d)$ ラウンドの定数 $\kappa$、ゲート所要時間、未構成の変換）のもとでの予備的な資源評価**であり、Kobayashi 方式の実装上の優位性を示す確定的な結果ではない。研究上の主な成果は、単一のコスト比よりも、未確定パラメータに対する**損益分岐と感度**（III-4, III-4b）である。

### 比較の約束
* 出力：両方式とも surface-code 系の論理 $T^\dagger|+\rangle$ → Pauli フレームの $\bar X$ で $T|+\rangle$（両方式コスト0、同一扱い）。
* 距離：Litinski は距離 $d$ の回転表面符号パッチ、Kobayashi は FSC（入出力の符号）の距離が $d$ となる $L=d$。D4 相（非 Pauli 符号）の距離が $L$ であることは $L\le5$（`FULL_KL=True` で $L=6$）でのみ証明済み（検証 V5b）。図の横軸は $L$（= FSC の距離）と表記する。
* **計数ルール R1–R3 は両方式で同一**（埋め込みの共通ライブラリ）。時間は「各符号のラウンド数」と「物理時間（下界〜上界）」を分けて示し、異なる符号のラウンドを同一視しない。
* 値の種類を区別して表示：**counted**（明示的回路から計数）、**literature**（文献値）、**assumption**（$\kappa$ など仮定パラメータに依存）、**unresolved**（未解決：除外、または例示）。

### 理想条件の限界
物理誤り率 0 では、どちらの方式も理想入力から理想出力を得る。さらに完全な入力魔法状態が使えるなら 15-to-1 蒸留は不要である。したがってこの比較は**「ノイズのない世界での最適な $T$ ゲート実装法」を決めるものではなく**、指定した 2 つのプロトコルの回路構造と資源の比較である。`FT_ref` のラウンド数は「ノイズがあれば必要になる $d$ 依存のラウンド数」を文献の慣習（Litinski: 1τ = $d$ cycles）と仮定（Kobayashi: $\kappa d$）で与えたものであり、ノイズ下の性能（論理誤り率）は評価していない。
'''),
    code(SETUP),
    code(INSTALL),
    code(PLOTSTYLE),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'msd_lib.py'),
    md(r'''
## §0. パラメータと資源表の計算
* `D_LIST`：評価する符号距離（$L=d$）。`DIST_L_MAX`：I-2 で厳密 MILP により符号距離を求める最大の $L$。
* `DUR`：各原始ゲートの所要時間（既定は全て 1 単位＝**仮定**）。例：`DUR['CCZ'] = 3`。
* `PARAMS`：$\kappa_0,\kappa_{in},\kappa_{out}$（$O(d)$ の定数；**文献で未確定のため仮定**）、注入時間など。
* `QUICK=True` で小規模動作確認。

両方式の資源表はここで 1 回だけ計算し（約 1 分）、第I〜III部はすべてこの表を使う。
'''),
    code(r'''
QUICK = False
D_LIST = [3, 5, 7, 9, 11, 13, 15]
DIST_L_MAX = 8
if QUICK:
    D_LIST, DIST_L_MAX = [3, 5, 7, 9, 11], 4
DUR = dict(DUR_UNIT)
PARAMS = dict(DEFAULT_PARAMS)
print(PARAMS)
d4st, d4s, blocks = d4_resource_tables(D_LIST, DUR, PARAMS)
msdst, msds, msd_per_d = msd_resource_tables(D_LIST, DUR, PARAMS)
d4st.to_csv(f'{RESULTS}/d4_stage_costs.csv', index=False); d4s.to_csv(f'{RESULTS}/d4_summary.csv', index=False)
json.dump(dict(params=PARAMS, dur=DUR, blocks=blocks), open(f'{RESULTS}/d4_blocks.json', 'w'), indent=1)
msdst.to_csv(f'{RESULTS}/msd_stage_costs.csv', index=False); msds.to_csv(f'{RESULTS}/msd_summary.csv', index=False)
msd_per_d.to_csv(f'{RESULTS}/msd_per_d.csv', index=False)
def _load_validation(name):
    p = f'{RESULTS}/{name}'
    return json.load(open(p)) if os.path.exists(p) else None
vd4, vmsd = _load_validation('validation_d4.json'), _load_validation('validation_msd.json')
print('distances:', D_LIST)
'''),
    md(r'''
### §0.1 検証状況（validation ノートの結果を読み込み）
FAIL があれば、その項目に依存する比較結果は信頼できない。未検証項目は下に列挙する。結果ファイルが無い場合（このランタイムで validation ノートを実行していない場合）は「未取得」と表示する。リポジトリ同梱の `results/validation_*.json` をこのノートと同じフォルダの `results/` に置けば読み込まれる。
'''),
    code(r'''
rows = []
for name, v in [('D4_switching_validation', vd4), ('MSD_15to1_validation', vmsd)]:
    if v is None:
        rows.append(dict(notebook=name, checks=None, PASS=None, FAIL=None, not_verified='（未取得：validation ノートの結果ファイルが見つからない）'))
        continue
    res = v['results']
    rows.append(dict(notebook=v['notebook'], checks=len(res), PASS=sum(r['status'] == 'PASS' for r in res.values()),
                     FAIL=sum(r['status'] == 'FAIL' for r in res.values()), not_verified='; '.join(v['not_verified'])))
val_df = pd.DataFrame(rows); val_df
'''),
    md(r'''
# 第I部 Kobayashi 方式：D4 符号への code switching による $|T_L\rangle$ 生成

**目的**：Kobayashi–Zhu–Hsin [KZH] のプロトコル（folded surface code (FSC) → D4符号 → transversal $U$（論理 $T^\dagger$）→ FSC）を、境界付き正方格子上で**ゲートレベルの回路として構成**し、段階ごとの物理量子ビット数・ラウンド数・物理時間・時空間体積・ゲート内訳を数える。

**ここでは正しさの検証はしない**（→ `D4_switching_validation.ipynb`）。ここで数えた回路が正しい論理操作を実装していることは、検証ノートで別途確認する。



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
    md(r'''
## I-1. 格子・境界・安定化子の構成（[KZH] Fig. 4 の再現）

境界頂点で許される $X$ 型生成元は、「$X$ 部分が境界の $Z$ 条件（$\mathcal L_r$: $Z^r$、$\mathcal L_b$: $Z^b$、$\mathcal L_{rb}$: $Z^rZ^b, Z^g$）と可換になる色の組」として**自動導出**している（手で境界生成元を書き下していない）。結果は Fig. 4 と一致する：
$\mathcal L_r$ 上は $S^g, S^b$、$\mathcal L_b$ 上は $S^r, S^g$、左上角は $S^g$、$\mathcal L_{rb}$ 上は $S^rS^b$、左下角・右上角は無し。
'''),
    code(r'''
L_FIG = 9
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
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_lattice_and_stabilizers.png', dpi=150); plt.show()
'''),
    md(r'''
右図：$S^r_X$ の CZ$^{g,b}$ は頂点の右上の plaquette（$v$ = 頂点0）、$S^g_X$ の CZ$^{r,b}$ は左上（$v$ = 頂点1）と右下（$v$ = 頂点2）、$S^b_X$ の CZ$^{r,g}$ は左下（$v$ = 頂点3）。これは [KZH] Fig. 4 の破線の位置と一致する。

## I-2. 符号データ：生成元数・論理量子ビット数・符号距離

* $k$：$\dim C_Z - \mathrm{rank}(\text{shifts})$（生成元の $C_Z$ 上での可換性・$S^2=1$ と併せて検証ノートで確認）。
* $d_Z$：論理 $Z$ 型 Pauli の最小重み。$d_X^{\rm shift}$：$C_Z$ に属し安定化子シフトの張る空間に属さない $X$ シフトの最小重み（D4相では CZ 修飾があるため、これは Knill–Laflamme 距離そのものではない。$L=2$ のみ KL 条件を直接検査）。
* いずれも厳密 MILP（HiGHS）。
'''),
    code(r'''
import time
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
code_df
'''),
    md(r'''
**結果**：調べた全ての $L$ で $k=1$、$d_Z = L$、$X$ 型シフト距離 $= L+3$（D4・FSC とも）。
* FSC（CSS 符号）の符号距離は厳密に $\min(d_Z,d_X)=L$。
* D4 相（非 Pauli 符号）では $d_Z=L$ の論理 $Z$ があるので距離 $\le L$。下界は検証ノート V5b で、重み $\le L-1$ の全 Pauli が Knill–Laflamme 条件を満たすことを $L\le5$（`FULL_KL=True` で $L=6$）について厳密に示した（$=L$）。それより大きな $L$ では D4 相の距離は**未証明**（上界 $L$ のみ）。
* よって比較の横軸は **$L$（= FSC の距離 = 入出力符号の距離）** であり、「D4 相も距離 $L$」は $L\le5$（6）でのみ証明済みである。$X$ 方向には $3$ だけ余分な距離があり、正方形でない格子で量子ビット数を削れる可能性があるが、[KZH] の Fig. 4 の正方形配置に従う（最適化はしていない）。

## I-3. 段階別の資源計数

資源表（§0 で計算済み）の元になる `D4CostData(L)` は、明示的な格子上で、各段階の回路を**ゲート単位で構成**して数える：
* ラウンド回路：各生成元に測定補助1個。$X$ 型 Clifford 安定化子 $S = X(s)\prod CZ$ は「補助 $|+\rangle$ → CNOT（補助→データ） → CCZ（補助, $i$, $j$） → $X$ 測定」（制御-$S$ による標準的な測定。**この回路は [KZH] には書かれておらず本研究の構成**。正しさは検証ノートで密行列により確認）。
* frozen 量子ビット（境界の単一量子ビット $Z$ 安定化子で $|0\rangle$ に固定）は割り当てず、それに作用するゲート（恒等になる）も除く。$\mathcal L_{rb}$ の $g$ 量子ビットを除くため、そこでの $V$ の CNOT 対（$Z^rZ^b=+1$ 上で恒等）も除く。
* S3 の CZ 補正は最悪ケース（全頂点）で数える（期待値はその半分；$\kappa$ が一様であることは検証ノートで統計的に確認）。
'''),
    md(r'''
### I-3.1 段階別の表（$d=9$、`FT_ref`、比較A、原始ゲートセット、データ＋測定補助）
`rounds` は符号ごとのラウンド数（S0, S4 は FSC のラウンド、S1 は D4 のラウンド）。`time_lb/ub` は物理時間（全ゲート1単位の仮定）の下界／上界。`source` は値の出所（文献／明示的回路からの計数／仮定パラメータ／未解決）。
'''),
    code(r'''
d_show = 9 if 9 in D_LIST else D_LIST[-1]
view = d4st[(d4st.d == d_show) & (d4st.model == 'FT_ref') & (d4st.comparison == 'A') &
                (d4st.gateset == 'primitive') & (~d4st.data_only)]
view[['stage', 'qubits', 'rounds', 'time_lb', 'time_ub', 'gates', 'source']]
'''),
    md(r'''
### I-3.2 1ラウンドあたりのコストと定数深さ段階
'''),
    code(r'''
rows = []
for d in D_LIST:
    b = blocks[f'{d}|primitive']; bd = blocks[f'{d}|clifford_t']
    rows.append(dict(d=d, FSC_round_lb=b['round']['FSC']['time_lb'], FSC_round_ub=b['round']['FSC']['time_ub'],
                     D4_round_lb=b['round']['D4']['time_lb'], D4_round_ub=b['round']['D4']['time_ub'],
                     D4_round_ub_cliffordT=bd['round']['D4']['time_ub'], D4g_round_ub=b['round']['D4g']['time_ub'],
                     CCZ_per_D4_round=b['round']['D4']['gates'].get('CCZ', 0),
                     CNOT_per_D4_round=b['round']['D4']['gates'].get('CNOT', 0),
                     U_depth_time=b['U_time'], switch_out_time=b['out_time'],
                     U_CS=b['U_gates'].get('CS', 0) + b['U_gates'].get('CSDG', 0), U_TDG=b['U_gates'].get('TDG', 0)))
per_round_df = pd.DataFrame(rows); per_round_df
'''),
    md(r'''
## I-4. 符号距離に対するスケーリング（厳密な多項式の同定）

数値はすべて明示的な格子上の計数なので、$L$ の多項式になることが期待される。最初の4点で補間した（高々3次の）多項式が**残りすべての点で厳密に一致するか**を確認する（一致しなければ「多項式ではない／境界の効果がまだある」と表示）。これは評価範囲の $L$ についての恒等式の確認であり、全ての $L$ についての証明ではない（ただし量子ビット数・ゲート数は格子の局所構造から $L$ の2次式、体積は $\lceil\kappa d\rceil$ ラウンドを掛けて3次式になる）。
'''),
    code(r'''
import sympy
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
        sel = d4s[(d4s.model == model) & (d4s.comparison == comp) & (d4s.gateset == 'primitive') & (~d4s.data_only)]
        quant[f'V [qubit*round] {model}'] = dict(zip(sel.d, sel.V_qubit_round))
        quant[f'V [qubit*time] lb {model}'] = dict(zip(sel.d, sel.V_qubit_time_lb))
        quant[f'V [qubit*time] ub {model}'] = dict(zip(sel.d, sel.V_qubit_time_ub))
        quant[f'rounds {model}'] = dict(zip(sel.d, sel.rounds))
poly_rows = []
for k, v in quant.items():
    p, ok = identify({d: int(round(val)) for d, val in v.items()})
    poly_rows.append(dict(quantity=k, polynomial_in_d=str(p), exact_on_all_points=ok))
poly_df = pd.DataFrame(poly_rows); poly_df.to_csv(f'{RESULTS}/d4_polynomials.csv', index=False); poly_df
'''),
    md(r'''
`FT_ref` の体積の多項式は $\kappa_0=\kappa_{in}=\kappa_{out}=1$（**仮定**）に対するもの。一般の $\kappa$ では、主要項は
$$V_{\rm qubit\cdot round}\simeq \big(8\kappa_0 + 12\kappa_{in} + 8\kappa_{out}\big)\,d^3$$
（FSC 相の量子ビット数 $=8d^2+4d-1$、D4 相 $=12d^2+4d-1$、いずれもデータ＋測定補助）。

## I-5. 図：段階別の内訳と量子ビット数
'''),
    code(r'''
sel = d4st[(d4st.d == d_show) & (d4st.model == 'FT_ref') & (d4st.comparison == 'A') &
               (d4st.gateset == 'primitive') & (~d4st.data_only)].copy()
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
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_qubits_vs_d.png', dpi=150); plt.show()
'''),
    md(r'''
## I-6. 解釈と注意

* **数えたもの**：明示的な格子・回路から数えた量子ビット数、ゲート数、ラウンドあたりの物理時間（下界・上界）。
* **仮定に依存するもの**：`FT_ref` のラウンド数（$\kappa$）。[KZH] は $O(d)$ としか述べておらず、JIT decoder の文献 [Dav] の全文は本環境から参照できなかった。ゲート所要時間（全て1単位）。
* **数えていないもの**：JIT decoder／matching・RG decoder の古典計算時間、制御系の遅延、FSC→通常パッチ変換（比較Bで未解決として扱う）。
* D4 相の 1 ラウンドは CCZ を含むため、FSC や回転表面符号の 1 ラウンドより深い（上の表）。**異なる符号のラウンド数を同一の時間単位とみなさない**ため、比較では物理時間でも示す。
* `FT_ref` では物理 CCZ が $6(d-1)^2$ 個／ラウンド × $\lceil\kappa_{in}d\rceil$ ラウンド $=\Theta(d^3)$ 必要になる。ノイズなしでは問題にならないが、ノイズありの拡張では重要な資源になる（比較Bで T 数としても表示）。
'''),
    md(r'''
# 第II部 Litinski 方式：15-to-1 magic state distillation

**目的**：Litinski *A Game of Surface Codes* [Lit] Sec. 3 の 15-to-1 プロトコル（Fig. 15 の回路を Fig. 17c の 11 tiles 配置で実行）を、
1. **tile モデル**（[Lit] の計数：1 tile = $d^2$ データ量子ビット（測定量子ビット込みで $2d^2$）、1 time step (τ) = $d$ code cycles）、
2. **物理モデル**（11 回の多パッチ測定 $P\otimes Z_m$ を Fig. 17c の配置上で**検査レベルまで明示的に構成**し（正しさは検証ノート M9）、各回転の量子ビット数・CNOT 数・ラウンドの物理時間（D4 側と同じスケジューラ）を数える。$Z_m\otimes Y_0$ の twist 継ぎ目の検査だけは構成していない）
の2通りで計数する。正しさの検証は `MSD_15to1_validation.ipynb`。

**15-to-1 の抽象回路（Reed–Muller 符号上の回路）だけを数えて lattice surgery の物理コストと同一視することはしない**：ここで数えるのは Fig. 17c の配置上の操作である。

### 段階（Fig. 17c）
| 段階 | 内容 | 時間 | 出所 |
|---|---|---|---|
| L0 | （比較Bのみ）qubits 1–4 への $|m\rangle$ 注入（並列） | $t_{\rm inj}$ | [Lit] App. A（Li の方法、3+3 cycles、$d$ に依存しない） |
| L1 | qubit 5 を $|+\rangle$ に初期化 | 0τ | [Lit] Sec. 1 |
| L2.k | auto-corrected π/8 回転 k（k=1..11）：$P\otimes Z_m$ と $Z_m\otimes Y_0$ の格子手術 | 1τ = $d$ ラウンド | [Lit] Fig. 17b,c |
| L2.ka | （比較Bのみ）回転 k 用の $|m\rangle$ 注入（11-tile 配置では直列） | $t_{\rm inj}$ | [Lit] Sec. 3.3 "Caveat" |
| L2.kb | $|0\rangle$ の X/Z 測定、$|m\rangle$ の X 測定 | 0τ（測定1層） | [Lit] |
| L3 | qubits 1–4 の X 測定 | 0τ | [Lit] |
| L4 | 論理 $\bar X$（Pauli フレーム）で $T^\dagger|+\rangle\to T|+\rangle$ | 0 | 本研究（D4 側と同じ扱い） |

**比較A**：入力 $|m\rangle$ は供給済み（L0, L2.ka を含めない）。**比較B**：注入時間と物理 T ゲート（15個）を含める。古典処理時間は両比較とも 0（**仮定**）。
'''),
    md(r'''
## II-1. 論理回路（式(1) → Fig. 15）
式(1)の行列 $M$ の列ごとに $Z$ 型 π/8 回転を置く。1列目〜4列目（重み1）は qubits 4,3,2,1 の初期状態 $|m\rangle$ に吸収され、残り 11 列が Fig. 15 の 11 回転になる（[Lit] Sec. 3.1）。
'''),
    code(r'''
rots = fig15_rotations()
pd.DataFrame([{'rotation': k + 1, **{f'q{i + 1}': ('Z' if i in r else '1') for i in range(5)}} for k, r in enumerate(rots)])
'''),
    md(r'''
## II-2. tile モデル（Fig. 17c の再計数）
配置を 3×5 の格子として転記した（`FIG17C_*`）。灰色の未使用 tile は数えない。
'''),
    code(r'''
sched = tile_schedule()
used = tiles_used(sched)
n_tiles = len(used); n_tau = sum(t for _, t, _ in sched)
print(f'tiles used = {n_tiles}, time steps = {n_tau} tau, tile*tau = {n_tiles * n_tau}')
print(f'leading-order space-time cost: {n_tiles * n_tau} d^3 (data qubits x cycles; [Lit] 121 d^3)'
      f' / {2 * n_tiles * n_tau} d^3 (incl. measurement qubits, 2d^2 per tile, [Lit] Sec. 4.3)')
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
colors = {'q1': C_LIT, 'q2': C_LIT, 'q3': C_LIT, 'q4': C_LIT, 'q5': C_AUX, 'm': '#eda100', 'zero': '#898781', 'anc': '#e4e3df'}
for ax, (title, lay) in zip(axes, [('Step 1 (0 tau)', FIG17C_STEP1), ('rotation step (1 tau each, x11)', FIG17C_ROTATION),
                                   ('after rotation / readout (0 tau)', FIG17C_FINAL)]):
    for i, row in enumerate(lay):
        for j, lab in enumerate(row):
            fc = colors.get(lab, '#fcfcfb') if lab else '#f0efec'
            ax.add_patch(plt.Rectangle((j, 2 - i), 0.96, 0.96, fc=fc, ec='#52514e', lw=0.8))
            if lab: ax.text(j + 0.48, 2 - i + 0.48, {'zero': '|0>', 'm': '|m>', 'anc': 'ancilla'}.get(lab, lab), ha='center', va='center', fontsize=9)
    ax.set_xlim(0, 5); ax.set_ylim(0, 3); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10)
plt.suptitle(f'Fig. 17c layout as transcribed: {n_tiles} tiles used', fontsize=11)
plt.tight_layout(); plt.savefig(f'{FIGURES}/msd_fig17c_layout.png', dpi=150); plt.show()
'''),
    md(r'''
## II-3. tile → 物理量子ビット・cycle への対応

| 量 | tile モデル | 物理モデル（本研究、明示的構成） |
|---|---|---|
| 1 tile | $d^2$ データ（[Lit] Sec. 1）、$2d^2$ 測定込み（[Lit] Sec. 4.3） | $d\times d$ データ量子ビットの領域 |
| qubits 1–5 | 1 tile ずつ | 回転表面符号パッチ。補助帯に面する辺を「$Z$ 辺」（重み2 の $X$ 検査）にする |
| $|m\rangle$ | 2 tiles | 2 tile の縦長パッチ。角を移動（[Lit] Fig. 2b）し、右辺下半分（補助帯側）と左辺（$|0\rangle$ 側）を $Z$ 辺にする |
| $|0\rangle$ | 2 tiles | 縦長の回転表面符号パッチ |
| 補助帯 | 2 tiles | 全量子ビットを $|+\rangle$ に初期化し、補助帯に触れる plaquette（合併領域に制限）を新たに測る。参加しないパッチに面する辺・自由辺は「$X$ 辺」（重み2 の $Z$ 検査）、両側が自由な角の量子ビットは $Z$ 測定 |
| 合併中に測る検査 | — | 新しい検査 $N$ ＋ $N$ と可換な旧パッチ検査 ＋ 可換部分群を完成させる旧検査の積（すべて重み ≤ 4、$\mathbb F_2$ 計算で決定） |
| 1τ | $d$ code cycles | $d$ ラウンド（`FT_ref`）／1 ラウンド（`NL_min`） |
| 1 ラウンドの物理時間 | — | 各回転の検査集合を D4 側と同じ可換クラス・スケジューラで計算（下界・上界） |

**構成していない部分**：$|m\rangle$ と $|0\rangle$ の間の $Z_m\otimes Y_0$ 測定（twist 欠陥を含む継ぎ目、$O(d)$ 個の測定量子ビット）。`source` 列と第III部の「未計上コスト」に明記する。
'''),
    code(r'''
msd_per_d    # per-d physical counts (computed in §0)
'''),
    md(r'''
### II-3.1 要約（`FT_ref`、データ＋測定量子ビット）
'''),
    code(r'''
cols = ['d', 'comparison', 'level', 'peak_qubits', 'rounds', 'time_lb', 'time_ub', 'V_qubit_round', 'V_qubit_time_lb', 'V_qubit_time_ub', 'nonclifford']
msds[(msds.model == 'FT_ref') & (~msds.data_only) & (msds.gateset == 'primitive')][cols]
'''),
    md(r'''
## II-4. スケーリング（厳密な多項式の同定）と [Lit] の値の再現
'''),
    code(r'''
import sympy
x = sympy.Symbol('d')
def identify(values):
    Ls = sorted(values)
    p = sympy.interpolate([(L, values[L]) for L in Ls[:4]], x)
    return sympy.expand(p), all(sympy.nsimplify(p.subs(x, L) - values[L]) == 0 for L in Ls)
rows = []
for level in ['tile', 'physical']:
    for data_only in [True, False]:
        for comp in ['A', 'B']:
            for model in ['FT_ref', 'NL_min']:
                sel = msds[(msds.level == level) & (msds.data_only == data_only) & (msds.comparison == comp) &
                             (msds.model == model) & (msds.gateset == 'primitive')]
                for q in ['V_qubit_round', 'peak_qubits', 'rounds']:
                    p, ok = identify({int(dd): int(round(v)) for dd, v in zip(sel.d, sel[q])})
                    rows.append(dict(level=level, data_only=data_only, comparison=comp, model=model, quantity=q,
                                     polynomial=str(p), exact_on_all_points=ok))
poly = pd.DataFrame(rows); poly.to_csv(f'{RESULTS}/msd_polynomials.csv', index=False)
poly[(poly.model == 'FT_ref') & (poly.quantity == 'V_qubit_round')]
'''),
    md(r'''
tile モデル・データのみ・比較A で $V = 121\,d^3$（[Lit] の値）、測定量子ビット込みで $242\,d^3$、物理モデル（明示的構成）では $242d^3-66d$（各回転の量子ビット数は $22d^2-6$）となる。旧版の近似（$22d^2-5$）との差は 1 量子ビット／回転で、ラウンドの物理時間も同じ（下界 6、上界 12）だった。比較Bでは注入時間 $12\,t_{\rm inj}$ cycles 分（$d$ に依存しない）が加わる。

## II-5. 図：時間に沿った量子ビット使用数（$d=9$、`FT_ref`、物理モデル、比較B）
'''),
    code(r'''
d_show = 9 if 9 in D_LIST else D_LIST[-1]
sel = msdst[(msdst.d == d_show) & (msdst.model == 'FT_ref') & (msdst.comparison == 'B') &
               (msdst.level == 'physical') & (~msdst.data_only) & (msdst.gateset == 'primitive')]
t, xs, ys = 0, [], []
for _, r in sel.iterrows():
    xs += [t, t + r.rounds]; ys += [r.qubits, r.qubits]; t += r.rounds
fig, ax = plt.subplots(figsize=(9, 3))
ax.fill_between(xs, ys, step=None, color=C_LIT, alpha=0.25, lw=0)
ax.plot(xs, ys, color=C_LIT)
ax.set_xlabel('time [rounds = surface-code cycles]'); ax.set_ylabel('active physical qubits')
ax.set_title(f'15-to-1 (Fig. 17c), d={d_show}: occupancy; area = qubit·round volume (comparison B incl. injection)')
plt.tight_layout(); plt.savefig(f'{FIGURES}/msd_occupancy_timeline.png', dpi=150); plt.show()
'''),
    md(r'''
## II-6. 解釈と注意
* tile モデルは [Lit] の値（11 tiles, 11τ, $121d^3$）を再現する（検証ノートで確認）。
* 物理モデル（明示的構成）との差は $O(d)$ の下位項のみ。ただし $Z_m\otimes Y_0$ の twist 継ぎ目の検査（$O(d)$ 個）は未計上。
* **含めていない費用**：古典処理（デコード・測定結果に応じた基底選択）の時間、Fig. 18 の追加 2×2 ブロックによる注入の並列化（tile 数が増える代わりに注入待ちが消える）、$Z_m\otimes Y_0$ の twist 欠陥の詳細な安定化子配置。
* ノイズが無い場合は蒸留そのものが不要（理想入力 $|m\rangle$ がそのまま出力品質）であり、ここでの比較は**回路構造と資源の比較**である（冒頭「理想条件の限界」）。
'''),
    md(r'''
# 第III部 比較（同じ計数ルール・同じ資源表）
以下はすべて §0 の資源表から計算する。
'''),
    md(r'''
## III-1. 段階別の資源コスト一覧（$d=9$、`FT_ref`、比較A、データ＋測定量子ビット、原始ゲートセット）
'''),
    code(r'''
d_show = 9 if 9 in D_LIST else D_LIST[len(D_LIST) // 2]
a = d4st[(d4st.d == d_show) & (d4st.model == 'FT_ref') & (d4st.comparison == 'A') & (d4st.gateset == 'primitive') & (~d4st.data_only)]
b = msdst[(msdst.d == d_show) & (msdst.model == 'FT_ref') & (msdst.comparison == 'A') & (msdst.gateset == 'primitive') &
          (~msdst.data_only) & (msdst.level == 'physical')]
# compress Litinski's 11 identical rotations into one row (x11)
def compress(df):
    out = []; rot = df[df.stage.str.contains('auto-corrected')]; rd = df[df.stage.str.contains('readout')]
    for _, r in df.iterrows():
        if 'auto-corrected' in r.stage or 'readout' in r.stage: continue
        out.append(r.to_dict())
    for sub, name in [(rot, 'L2 auto-corrected pi/8 rotation (x11)'), (rd, 'L2b readout |0>,|m> (x11)')]:
        if len(sub):
            r = sub.iloc[0].to_dict(); r.update(stage=name, rounds=sub.rounds.sum(), time_lb=sub.time_lb.sum(), time_ub=sub.time_ub.sum())
            out.append(r)
    return pd.DataFrame(out)
tab = pd.concat([a.assign(V_qubit_round=a.qubits * a.rounds), compress(b).assign(V_qubit_round=lambda x: x.qubits * x.rounds)])
tab = tab[['protocol', 'stage', 'qubits', 'rounds', 'time_lb', 'time_ub', 'V_qubit_round', 'source']]
tab.to_csv(f'{RESULTS}/comparison_stage_table_d{d_show}.csv', index=False)
tab
'''),
    md(r'''
`rounds` は Kobayashi では FSC／D4 のラウンド、Litinski では回転表面符号のラウンド。単位が違うので、横並びにする量は物理時間（`time_lb/ub`、全ゲート1単位の仮定）である。

## III-2. 主比較：$d$ 依存性（`FT_ref`、比較A、データ＋測定量子ビット）
'''),
    code(r'''
def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items(): m &= (df[k] == v).values
    return df[m].sort_values('d')
K = pick(d4s, model='FT_ref', comparison='A', gateset='primitive', data_only=False)
Lp = pick(msds, model='FT_ref', comparison='A', gateset='primitive', data_only=False, level='physical')
Lt = pick(msds, model='FT_ref', comparison='A', gateset='primitive', data_only=False, level='tile')
main = pd.DataFrame(dict(d=K.d.values,
    K_peak_qubits=K.peak_qubits.values, L_peak_qubits=Lp.peak_qubits.values,
    K_rounds=K.rounds.values, L_rounds=Lp.rounds.values,
    K_time_lb=K.time_lb.values, K_time_ub=K.time_ub.values, L_time_lb=Lp.time_lb.values, L_time_ub=Lp.time_ub.values,
    K_V_qr=K.V_qubit_round.values, L_V_qr=Lp.V_qubit_round.values, L_V_qr_tile_lit=Lt.V_qubit_round.values,
    K_V_t_lb=K.V_qubit_time_lb.values, K_V_t_ub=K.V_qubit_time_ub.values, L_V_t_lb=Lp.V_qubit_time_lb.values, L_V_t_ub=Lp.V_qubit_time_ub.values))
main['ratio_peak_qubits'] = main.K_peak_qubits / main.L_peak_qubits
main['ratio_V_qubit_round'] = main.K_V_qr / main.L_V_qr
main['ratio_V_time_lb'] = main.K_V_t_lb / main.L_V_t_lb
main['ratio_V_time_ub'] = main.K_V_t_ub / main.L_V_t_ub
main['ratio_V_time_range'] = [f'{min(x, y):.3f}–{max(x, y):.3f}' for x, y in
                              zip(main.K_V_t_lb / main.L_V_t_ub, main.K_V_t_ub / main.L_V_t_lb)]
main.to_csv(f'{RESULTS}/comparison_main_FT_ref_A.csv', index=False)
main.round(3)
'''),
    md(r'''
* `ratio_V_time_lb` は両方式の下界同士、`ratio_V_time_ub` は上界同士の比。`ratio_V_time_range` は最も保守的な範囲（K下界/L上界 〜 K上界/L下界）。
* Kobayashi の値は $\kappa_0=\kappa_{in}=\kappa_{out}=1$（**仮定**）に依存する。下の III-4 で $\kappa$ 依存性を示す。
'''),
    code(r'''
ds = main.d.values
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
ax = axes[0, 0]
ax.plot(ds, main.K_peak_qubits, 'o-', color=C_KOB, label='Kobayashi (counted)')
ax.plot(ds, main.L_peak_qubits, 's-', color=C_LIT, label='Litinski 15-to-1 (counted, physical model)')
ax.set_title('Peak physical qubits (data + measurement)'); ax.set_xlabel('L (= FSC code distance d)'); ax.set_ylabel('qubits'); ax.legend()
ax = axes[0, 1]
ax.fill_between(ds, main.K_time_lb, main.K_time_ub, color=C_KOB, alpha=0.18, lw=0)
ax.plot(ds, main.K_time_lb, 'o--', color=C_KOB, label='Kobayashi, kappa=1 (assumption): lower / upper bound')
ax.plot(ds, main.K_time_ub, 'o--', color=C_KOB, alpha=0.6)
ax.fill_between(ds, main.L_time_lb, main.L_time_ub, color=C_LIT, alpha=0.18, lw=0)
ax.plot(ds, main.L_time_lb, 's-', color=C_LIT, label='Litinski, 1 tau = d cycles (literature): lower / upper bound')
ax.plot(ds, main.L_time_ub, 's-', color=C_LIT, alpha=0.6)
ax.set_title('Execution time (physical time units; all gates = 1 unit)'); ax.set_xlabel('L (= FSC code distance d)'); ax.set_ylabel('time units'); ax.legend(fontsize=8)
ax = axes[1, 0]
ax.fill_between(ds, main.K_V_t_lb, main.K_V_t_ub, color=C_KOB, alpha=0.18, lw=0)
ax.plot(ds, main.K_V_t_lb, 'o--', color=C_KOB, label='Kobayashi, kappa=1 (assumption)')
ax.plot(ds, main.K_V_t_ub, 'o--', color=C_KOB, alpha=0.6)
ax.fill_between(ds, main.L_V_t_lb, main.L_V_t_ub, color=C_LIT, alpha=0.18, lw=0)
ax.plot(ds, main.L_V_t_lb, 's-', color=C_LIT, label='Litinski (counted, physical model)')
ax.plot(ds, main.L_V_t_ub, 's-', color=C_LIT, alpha=0.6)
ax.set_yscale('log'); ax.set_title('Space-time volume [qubit x time unit] (band = lower..upper bound)')
ax.set_xlabel('L (= FSC code distance d)'); ax.legend(fontsize=8)
ax = axes[1, 1]
ax.plot(ds, main.ratio_V_qubit_round, 'o--', color=C_KOB, label='V [qubit x own rounds] (kappa=1 assumed)')
ax.plot(ds, main.ratio_V_time_lb, '^--', color=C_AUX, label='V [qubit x time], lower bounds')
ax.plot(ds, main.ratio_V_time_ub, 'v--', color='#6250d6', label='V [qubit x time], upper bounds')
ax.plot(ds, main.ratio_peak_qubits, 'D-', color='#52514e', label='peak qubits (counted)')
ax.axhline(1, color='#898781', lw=1)
ax.set_ylim(0, max(1.1, main[['ratio_V_qubit_round', 'ratio_V_time_lb', 'ratio_V_time_ub', 'ratio_peak_qubits']].values.max() * 1.1))
ax.set_title('Ratio Kobayashi / Litinski (<1: Kobayashi cheaper)'); ax.set_xlabel('L (= FSC code distance d)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_main_FT_ref_A.png', dpi=150); plt.show()
'''),
    md(r'''
## III-3. 先頭係数（漸近的スケーリング）
最初の 4 点で補間した多項式が全ての点で厳密に一致するかを確認する（$\kappa=1$、`FT_ref`、比較A、データ＋測定量子ビット）。
'''),
    code(r'''
rows = []
sel = {('Kobayashi', 'FT_ref'): K, ('Litinski physical', 'FT_ref'): Lp, ('Litinski tile', 'FT_ref'): Lt}
for (name, model), df in sel.items():
    for q in ['peak_qubits', 'rounds', 'V_qubit_round', 'V_qubit_time_lb', 'V_qubit_time_ub']:
        p, ok = identify_polynomial(dict(zip(df.d, df[q])))
        rows.append(dict(protocol=name, quantity=q, polynomial_in_d=str(p), exact_on_all_points=ok))
poly_df = pd.DataFrame(rows); poly_df.to_csv(f'{RESULTS}/comparison_polynomials.csv', index=False); poly_df
'''),
    md(r'''
まとめると（データ＋測定量子ビット、`FT_ref`、比較A）：
* Kobayashi：$V_{\rm qubit\cdot round} = (8\kappa_0+12\kappa_{in}+8\kappa_{out})d^3 + O(d^2)$ → $\kappa=1$ で $28d^3+\dots$
* Litinski：$V_{\rm qubit\cdot round} = 242d^3 - 66d$（物理モデル＝明示的構成）、$242d^3$（tile モデル、測定量子ビット込み）、$121d^3$（tile、データのみ＝[Lit] の値）
* ピーク量子ビット：Kobayashi $12d^2+4d-1$、Litinski $22d^2-6$（物理モデル＝明示的構成）

ただし Kobayashi の 1 ラウンド（D4 相）は物理時間で回転表面符号の 1 ラウンドより長い（上界 28 vs 12、下界 9 vs 6、全ゲート1単位）。物理時間で測った体積の比は III-2 の表を参照。

## III-4. $\kappa$（$O(d)$ の定数）への依存性と損益分岐
[KZH] は switch-in／EC のラウンド数を $O(d)$ としか述べていないため、$\kappa$ を振る。D4 の構成要素（`blocks`）から体積を再構成する（$\kappa=1$ で資源表と一致することを確認）。
'''),
    code(r'''
import math
def k_volume(d, k0, kin, kout, gs='primitive'):
    b = blocks[f'{d}|{gs}']; n = b['n']
    nf = n['fsc_data'] + n['fsc_anc']; n4 = n['d4_data'] + n['d4_anc']
    r0, r1, r4 = math.ceil(k0 * d), math.ceil(kin * d), math.ceil(kout * d)
    Rf, R4 = b['round']['FSC'], b['round']['D4']
    vqr = nf * r0 + n4 * r1 + nf * r4
    const = n['d4_data'] * (b['U_time'] + b['out_time']) + nf * b['xcorr_time']
    vlb = nf * (r0 + r4) * Rf['time_lb'] + n4 * r1 * R4['time_lb'] + const
    vub = nf * (r0 + r4) * Rf['time_ub'] + n4 * r1 * R4['time_ub'] + const
    return vqr, vlb, vub
for d in D_LIST:
    v = k_volume(d, 1, 1, 1); row = K[K.d == d].iloc[0]
    assert (v[0], v[1], v[2]) == (row.V_qubit_round, row.V_qubit_time_lb, row.V_qubit_time_ub), (d, v)
print('reconstruction from blocks reproduces the CSV at kappa=1: OK')
kins = np.linspace(0.25, 30, 240)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dd = D_LIST[-1]
Lrow = Lp[Lp.d == dd].iloc[0]
for ax, (k0, kout, title) in zip(axes, [(1, 1, 'kappa0 = kappa_out = 1'), (0, 1, 'kappa0 = 0 (init merged, optimistic), kappa_out = 1')]):
    vs = np.array([k_volume(dd, k0, k, kout) for k in kins])
    ax.plot(kins, vs[:, 0] / Lrow.V_qubit_round, '--', color=C_KOB, label='V[qubit x own rounds]')
    ax.plot(kins, vs[:, 1] / Lrow.V_qubit_time_lb, '--', color=C_AUX, label='V[qubit x time], lower/lower')
    ax.plot(kins, vs[:, 2] / Lrow.V_qubit_time_ub, '--', color='#6250d6', label='V[qubit x time], upper/upper')
    ax.axhline(1, color='#898781', lw=1)
    ax.set_xlabel('kappa_in  (switch-in rounds = ceil(kappa_in * d))'); ax.set_ylabel('Kobayashi / Litinski')
    ax.set_title(f'd = {dd}: {title}', fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_kappa_sweep.png', dpi=150); plt.show()
rows = []
for d in D_LIST:
    Lr = Lp[Lp.d == d].iloc[0]
    for k0, kout in [(1, 1), (0, 1), (2, 2)]:
        be = {}
        for j, key in enumerate(['V_qubit_round', 'V_qubit_time_lb', 'V_qubit_time_ub']):
            ks = [k for k in np.arange(0.1, 200, 0.1) if k_volume(d, k0, k, kout)[j] >= Lr[key]]
            be[key] = round(ks[0], 1) if ks else None
        rows.append(dict(d=d, kappa0=k0, kappa_out=kout, breakeven_kappa_in_Vqr=be['V_qubit_round'],
                         breakeven_kappa_in_Vtime_lb=be['V_qubit_time_lb'], breakeven_kappa_in_Vtime_ub=be['V_qubit_time_ub']))
be_df = pd.DataFrame(rows); be_df.to_csv(f'{RESULTS}/comparison_breakeven_kappa.csv', index=False); be_df
'''),
    md(r'''
損益分岐 $\kappa_{in}$：これより大きい定数で switch-in に $\kappa_{in}d$ ラウンドが必要なら、Kobayashi 方式の体積が Litinski 方式（比較A、物理モデル）を上回る。主要項だけで見ると $V_{\rm qubit\cdot round}$ では $8\kappa_0+12\kappa_{in}+8\kappa_{out}=242$ より $\kappa_{in}\approx 18.8$（$\kappa_0=\kappa_{out}=1$）。物理時間の体積では D4 ラウンドの方が深いので損益分岐はより小さい（表）。
'''),
    md(r'''
## III-4b. ゲート所要時間（CCZ）への感度
III-2 の物理時間は全ゲート 1 単位という仮定に依存する。D4 側のラウンドは CCZ を含むので、CCZ の所要時間 $t_{\rm CCZ}$ を 1〜12 単位で変え、$\kappa_{in}$ と合わせて体積比（物理時間、上界同士・下界同士）がどう変わるかを示す（Litinski 側は CCZ を使わないので不変）。
'''),
    code(r'''
t_list = [1, 2, 3, 5, 8, 12]
d_s = D_LIST[-1]
Lrow_s = Lp[Lp.d == d_s].iloc[0]
cds = {}
for tc in t_list:
    dur_ = dict(DUR_UNIT); dur_['CCZ'] = tc
    cds[tc] = D4CostData(d_s, dur_)
def k_volume_cd(cd, k0, kin, kout):
    n = cd.n; nf = n['fsc_data'] + n['fsc_anc']; n4 = n['d4_data'] + n['d4_anc']; d = cd.L
    r0, r1, r4 = math.ceil(k0 * d), math.ceil(kin * d), math.ceil(kout * d)
    const = n['d4_data'] * (cd.U['time'] + cd.out['time']) + nf * cd.x_corr['time']
    lb = nf * (r0 + r4) * cd.round['FSC']['time_lb'] + n4 * r1 * cd.round['D4']['time_lb'] + const
    ub = nf * (r0 + r4) * cd.round['FSC']['time_ub'] + n4 * r1 * cd.round['D4']['time_ub'] + const
    return lb, ub
assert k_volume_cd(cds[1], 1, 1, 1) == (K[K.d == d_s].iloc[0].V_qubit_time_lb, K[K.d == d_s].iloc[0].V_qubit_time_ub)
rows = []
for tc in t_list:
    lb, ub = k_volume_cd(cds[tc], 1, 1, 1)
    be = {}
    for j, key in enumerate(['V_qubit_time_lb', 'V_qubit_time_ub']):
        ks = [k for k in np.arange(0.1, 60, 0.1) if k_volume_cd(cds[tc], 1, k, 1)[j] >= Lrow_s[key]]
        be[key] = round(ks[0], 1) if ks else None
    rows.append(dict(d=d_s, t_CCZ=tc, D4_round_lb=cds[tc].round['D4']['time_lb'], D4_round_ub=cds[tc].round['D4']['time_ub'],
                     ratio_lb_kappa1=lb / Lrow_s.V_qubit_time_lb, ratio_ub_kappa1=ub / Lrow_s.V_qubit_time_ub,
                     breakeven_kappa_in_lb=be['V_qubit_time_lb'], breakeven_kappa_in_ub=be['V_qubit_time_ub']))
sens_df = pd.DataFrame(rows); sens_df.to_csv(f'{RESULTS}/comparison_sensitivity_tCCZ.csv', index=False)
kgrid = np.linspace(0.5, 20, 40)
Rm = np.array([[k_volume_cd(cds[tc], 1, k, 1)[1] / Lrow_s.V_qubit_time_ub for k in kgrid] for tc in t_list])
fig, ax = plt.subplots(figsize=(7, 3.8))
im = ax.pcolormesh(kgrid, np.arange(len(t_list)), Rm, cmap='Blues', shading='nearest', vmin=0, vmax=max(2, Rm.max()))
cs = ax.contour(kgrid, np.arange(len(t_list)), Rm, levels=[1.0], colors='#0b0b0b', linewidths=1.5)
ax.clabel(cs, fmt='ratio = 1', fontsize=8)
ax.set_yticks(range(len(t_list))); ax.set_yticklabels([str(t) for t in t_list])
ax.set_xlabel('kappa_in (switch-in rounds = ceil(kappa_in d)); kappa0 = kappa_out = 1')
ax.set_ylabel('t_CCZ [time units]')
ax.set_title(f'L = {d_s}: volume ratio Kobayashi/Litinski [qubit x time], upper bounds', fontsize=10)
plt.colorbar(im, ax=ax, label='ratio'); plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_sensitivity_kappa_tCCZ.png', dpi=150); plt.show()
sens_df.round(3)
'''),
    md(r'''
CCZ が遅いほど D4 のラウンドが長くなり、損益分岐の $\kappa_{in}$ は小さくなる。比 1 の等高線より左下が「Kobayashi 方式の体積が小さい」領域である（他の仮定はすべて既定値）。
'''),
    md(r'''
## III-5. ノイズなし最小モデル（`NL_min`）
繰り返し測定を一切しない論理的最小。**耐故障性の主張ではない**（冒頭「理想条件の限界」）。
'''),
    code(r'''
Kn = pick(d4s, model='NL_min', comparison='A', gateset='primitive', data_only=False)
Ln = pick(msds, model='NL_min', comparison='A', gateset='primitive', data_only=False, level='physical')
nl = pd.DataFrame(dict(d=Kn.d.values, K_peak=Kn.peak_qubits.values, L_peak=Ln.peak_qubits.values,
                       K_rounds=Kn.rounds.values, L_rounds=Ln.rounds.values, K_time_lb=Kn.time_lb.values, K_time_ub=Kn.time_ub.values,
                       L_time_lb=Ln.time_lb.values, L_time_ub=Ln.time_ub.values,
                       K_V_qr=Kn.V_qubit_round.values, L_V_qr=Ln.V_qubit_round.values,
                       ratio_V_time_lb=Kn.V_qubit_time_lb.values / Ln.V_qubit_time_lb.values,
                       ratio_V_time_ub=Kn.V_qubit_time_ub.values / Ln.V_qubit_time_ub.values))
nl.to_csv(f'{RESULTS}/comparison_NL_min_A.csv', index=False); nl.round(3)
'''),
    md(r'''
## III-6. 比較B：可能な範囲での完全な資源比較

追加で計上するもの
* Litinski：$|m\rangle$ 注入 15 回（各 $t_{\rm inj}=6$ cycles、[Lit] App. A の記述；11-tile 配置では回転ごとに直列）、物理 $T$ ゲート 15 個。
* Kobayashi：物理非 Clifford ゲート（CCZ：測定ガジェット、CS/CS$^\dagger$/$T^\dagger$：$U$）。Clifford+T 分解時の T 数と、分解でラウンドが深くなる効果。
* **FSC → 通常の回転表面符号パッチ変換：未解決**。これは**合計に 0 として含めず**、下の「未計上コスト」の表に分けて示す。したがって比較Bの Kobayashi の合計は変換を含まない**下界**である。参考として「1τ の格子手術（FSC＋2 tiles）」という**例示的仮定**を加えた値も別列に示す（検証された手順ではない）。
'''),
    code(r'''
unc = pd.DataFrame([dict(protocol=p, comparison=c, uncounted_cost=a, status=b)
                    for p in ['Kobayashi_D4', 'Litinski_15to1'] for c in ['A', 'B'] for a, b in uncounted_costs(p, c)])
unc.to_csv(f'{RESULTS}/uncounted_costs.csv', index=False); unc
'''),
    code(r'''
rows = []
for d in D_LIST:
    for gs in ['primitive', 'clifford_t']:
        k = pick(d4s, model='FT_ref', comparison='B', gateset=gs, data_only=False); k = k[k.d == d].iloc[0]
        l = pick(msds, model='FT_ref', comparison='B', gateset=gs, data_only=False, level='physical'); l = l[l.d == d].iloc[0]
        nf = blocks[f'{d}|{gs}']['n']; conv_q = nf['fsc_data'] + nf['fsc_anc'] + 4 * d * d
        rows.append(dict(d=d, gateset=gs, K_peak=k.peak_qubits, L_peak=l.peak_qubits,
                         K_time_lb=k.time_lb, K_time_ub=k.time_ub, L_time_lb=l.time_lb, L_time_ub=l.time_ub,
                         K_V_time_lb=k.V_qubit_time_lb, K_V_time_ub=k.V_qubit_time_ub,
                         L_V_time_lb=l.V_qubit_time_lb, L_V_time_ub=l.V_qubit_time_ub,
                         K_V_qr=k.V_qubit_round, K_V_qr_with_illustrative_conversion=k.V_qubit_round + conv_q * d,
                         L_V_qr=l.V_qubit_round,
                         K_nonclifford=k.nonclifford, L_nonclifford=l.nonclifford,
                         K_Tcount_if_decomposed=k.T_count_if_decomposed, L_Tcount=l.T_count_if_decomposed))
Bdf = pd.DataFrame(rows); Bdf.to_csv(f'{RESULTS}/comparison_B_FT_ref.csv', index=False)
Bdf[Bdf.gateset == 'primitive'][['d', 'K_peak', 'L_peak', 'K_V_qr', 'K_V_qr_with_illustrative_conversion', 'L_V_qr', 'K_nonclifford', 'L_nonclifford', 'K_Tcount_if_decomposed', 'L_Tcount']]
'''),
    code(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
Bp = Bdf[Bdf.gateset == 'primitive']; Bc = Bdf[Bdf.gateset == 'clifford_t']
ax = axes[0]
ax.plot(Bp.d, Bp.K_Tcount_if_decomposed, 'o--', color=C_KOB, label='Kobayashi: T-count if CCZ/CS decomposed (kappa=1 assumed)')
ax.plot(Bp.d, Bp.L_Tcount, 's-', color=C_LIT, label='Litinski: 15 physical T (injection)')
ax.set_yscale('log'); ax.set_xlabel('L (= FSC code distance d)'); ax.set_ylabel('physical non-Clifford (T) count')
ax.set_title('Physical non-Clifford resources (comparison B)'); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(Bc.d, Bc.K_V_time_ub / Bc.L_V_time_ub, 'v--', color='#6250d6', label='Clifford+T gate set, upper/upper')
ax.plot(Bc.d, Bc.K_V_time_lb / Bc.L_V_time_lb, '^--', color=C_AUX, label='Clifford+T gate set, lower/lower')
ax.plot(Bp.d, Bp.K_V_time_ub / Bp.L_V_time_ub, 'v:', color='#6250d6', alpha=0.6, label='native CCZ/CS, upper/upper')
ax.plot(Bp.d, Bp.K_V_time_lb / Bp.L_V_time_lb, '^:', color=C_AUX, alpha=0.6, label='native CCZ/CS, lower/lower')
ax.axhline(1, color='#898781', lw=1); ax.set_ylim(0, None)
ax.set_xlabel('L (= FSC code distance d)'); ax.set_ylabel('Kobayashi / Litinski')
ax.set_title('Volume ratio [qubit x time], comparison B (lower bound)'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_B.png', dpi=150); plt.show()
'''),
    md(r'''
## III-7. 仮定・未確定事項の一覧（表）
詳細な説明は `docs/02_assumptions_and_open_issues.md`。
'''),
    code(r'''
assump = pd.DataFrame([
    ('A1', 'Kobayashi switch-in rounds = ceil(kappa_in d), kappa_in=1', 'assumption', '[KZH] states only O(d); [Dav] full text not accessible here'),
    ('A2', 'Kobayashi FSC EC rounds after return = ceil(kappa_out d), kappa_out=1', 'assumption', '[KZH] O(d)'),
    ('A3', 'FSC |+_L> preparation rounds = ceil(kappa0 d), kappa0=1', 'assumption', 'not specified in [KZH]; Litinski counts patch init as 0 tau'),
    ('A4', 'Litinski 1 tau = d code cycles', 'literature', '[Lit] Sec. 1'),
    ('A5', 'Every primitive gate / init / measurement = 1 time unit', 'assumption', 'parametrised (DUR)'),
    ('A6', 'Clifford stabilizers measured by ancilla-controlled-S (CNOT + CCZ) gadgets', 'our construction (validated)', 'circuit not given in [KZH]; V9'),
    ('A7', 'Round schedule: commuting-class upper bound / busiest-qubit lower bound', 'our construction (validated)', 'V10; optimised interleavings could be shorter'),
    ('A8', 'Frozen boundary qubits not allocated; gates on them dropped', 'counting rule', 'same rule for both codes'),
    ('A9', 'Litinski multi-patch P(x)Z_m surgeries constructed explicitly (M9); Z_m(x)Y_0 twist seam not constructed', 'counted / partly uncounted', 'M9; O(d) twist-seam checks uncounted'),
    ('A10', 'Injection: 3+3 cycles per |m>, serial in the 11-tile layout', 'literature', '[Lit] App. A, Sec. 3.3 Caveat'),
    ('A11', 'Classical processing / decoding latency = 0', 'assumption', 'both protocols'),
    ('U1', 'FSC -> standard rotated patch conversion', 'UNRESOLVED', 'excluded in B (lower bound); illustrative 1-tau value shown separately'),
    ('U2', 'D4-phase Knill-Laflamme distance for L >= 6 (7 with FULL_KL)', 'UNVERIFIED', 'proven = L for L <= 5 (V5b), L = 6 with FULL_KL'),
    ('U3', 'Fault tolerance of the switching (JIT decoder) and value of kappa', 'UNRESOLVED', 'requires noisy simulation (next stage)'),
], columns=['id', 'item', 'status', 'basis / note'])
assump.to_csv(f'{RESULTS}/assumptions_table.csv', index=False); assump
'''),
    md(r'''
## III-8. 結論（理想条件下、指定した 2 プロトコルの回路構造に関する**予備的**評価）

1. **正しさ**：Kobayashi 方式は、境界付き D4 符号（Fig. 4）上で $U=WV$ が論理 $T^\dagger$ を実装し、ランダムな測定結果に対する補正（$Z^rZ^b$ の Pauli フレーム補正と $S^g_X$ の CZ 補正）を含めて FSC 上に $T^\dagger|+\rangle$ を生成する。根拠は、$L\le2$ の符号状態の厳密シミュレーション、$L\le4$ での回路と位相公式の照合、$L\le 8$ の位相公式の局所厳密証明（それぞれ意味の異なる検証）。D4 相の距離が $L$ であることは $L\le5$（6）で証明済み。Litinski 方式は 15-to-1 の論理動作・誤り検出・auto-corrected π/8・11 tiles × 11τ・$121d^3$ を再現し、11 回の多パッチ測定を物理レベルで明示的に構成・検証した（$Z_m\otimes Y_0$ の twist 継ぎ目を除く）。
2. **量子ビット数**（計数）：ピークで Kobayashi $12L^2+4L-1$ に対し Litinski $22d^2-6$（データ＋測定量子ビット、$L=d$）。
3. **時間・体積**（仮定に依存）：$\kappa=1$・全ゲート 1 単位では、Kobayashi は $3d$ ラウンド＋定数深さ、Litinski は $11d$ ラウンドで、体積比は約 0.1〜0.2。しかしこれは仮定に強く依存する：損益分岐 $\kappa_{in}$ は qubit·round 体積で約 15〜19、物理時間体積で約 6〜13（III-4）、さらに CCZ が遅いと小さくなる（III-4b）。**$O(d)$ の定数とゲート時間が確定するまで、優劣は確定しない**。
4. **比較B**：Kobayashi 方式は $\Theta(d^3)$ 個の物理 CCZ（Clifford+T 分解で $\approx 42d^3$ 個の T）を使うのに対し、Litinski 方式の物理非 Clifford 資源は T 15 個。FSC→通常パッチ変換は未解決で、合計には含めず「未計上コスト」として分離した。比較Bは**共通のスタートとゴールからの完全な比較としては未完成**である。
5. **限界**：ノイズ0では蒸留は本来不要であり、本比較は最適な T ゲート実装法を決めるものではない。論文の主張として使うには、ノイズ下での $\kappa$ の決定、変換回路の構成、ハードウェアに即したゲート時間が必要である。
'''),
]
save('resource_analysis.ipynb', cells)
