from nbutil import *

cells = [md(r'''
# resource_comparison — Kobayashi方式（D4 code switching）と Litinski方式（15-to-1）の資源比較

**このノートは単独で実行できる**（Google Colab ではノートブックごとに別ランタイムなので、他のノートの出力ファイルに依存しない）。両方式の資源表は、他の main ノートと同じライブラリ関数（下に埋め込み）でこのノート内で再計算する。検証結果（`validation_*.json`）は、同じフォルダ（例：Google Drive）で validation ノートを実行済みなら読み込んで表示し、無ければ「未取得」と表示する。

### 結果の位置づけ（最初に読むこと）
このノートの数値は、**特定の仮定（とくに $O(d)$ ラウンドの定数 $\kappa$、ゲート所要時間、未構成の変換）のもとでの予備的な資源評価**であり、Kobayashi 方式の実装上の優位性を示す確定的な結果ではない。研究上の主な成果は、単一のコスト比よりも、未確定パラメータに対する**損益分岐と感度**（§4, §4b）である。

### このノートでの比較の約束
* 出力：両方式とも surface-code 系の論理 $T^\dagger|+\rangle$ → Pauli フレームの $\bar X$ で $T|+\rangle$（両方式コスト0、同一扱い）。
* 距離：Litinski は距離 $d$ の回転表面符号パッチ、Kobayashi は FSC（入出力の符号）の距離が $d$ となる $L=d$。D4 相（非 Pauli 符号）の距離が $L$ であることは $L\le5$（`FULL_KL=True` で $L=6$）でのみ厳密に証明済み（検証 V5b）。図の横軸は $L$（= FSC の距離）と表記する。
* **計数ルール R1–R3 は両方式で同一**（埋め込みの共通ライブラリ）。時間は「各符号のラウンド数」と「物理時間（下界〜上界）」を分けて示し、異なる符号のラウンドを同一視しない。
* 値の種類を区別して表示：**counted**（明示的回路から計数）、**literature**（文献値）、**assumption**（$\kappa$ など仮定パラメータに依存）、**unresolved**（未解決：除外、または例示）。

### §3.3 の限界（必ず併読）
物理誤り率0では、どちらの方式も理想入力から理想出力を得る。さらに完全な入力魔法状態が使えるなら 15-to-1 蒸留は不要である。したがってこの比較は**「ノイズのない世界での最適な $T$ ゲート実装法」を決めるものではなく**、指定した 2 つのプロトコルの回路構造と資源の比較である。`FT_ref` のラウンド数は「ノイズがあれば必要になる $d$ 依存のラウンド数」を文献の慣習（Litinski: 1τ = $d$ cycles）と仮定（Kobayashi: $\kappa d$）で与えたものであり、ノイズ下の性能（論理誤り率）は評価していない。
'''),
    code(SETUP), code(INSTALL), code(PLOTSTYLE),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'msd_lib.py'),
    md(r'''## パラメータと資源表の計算
既定値は main ノートと同じ。`D_LIST`、`DUR`（ゲート所要時間）、`PARAMS`（$\kappa$ など）を変えるとこのノート内の全ての表・図が再計算される。約 1 分。'''),
    code(r'''D_LIST = [3, 5, 7, 9, 11, 13, 15]
DUR = dict(DUR_UNIT)
PARAMS = dict(DEFAULT_PARAMS)
d4st, d4s, blocks = d4_resource_tables(D_LIST, DUR, PARAMS)
msdst, msds, msd_per_d = msd_resource_tables(D_LIST, DUR, PARAMS)
def _load_validation(name):
    p = f'{RESULTS}/{name}'
    return json.load(open(p)) if os.path.exists(p) else None
vd4, vmsd = _load_validation('validation_d4.json'), _load_validation('validation_msd.json')
print('distances:', D_LIST)'''),
    md(r'''## 0. 検証状況（validation ノートの結果を読み込み）
FAIL があれば、その項目に依存する比較結果は信頼できない。未検証項目は下に列挙する。結果ファイルが無い場合（このランタイムで validation ノートを実行していない場合）は「未取得」と表示する。リポジトリ同梱の `results/validation_*.json` をこのノートと同じフォルダの `results/` に置けば読み込まれる。'''),
    code(r'''rows = []
for name, v in [('D4_switching_validation', vd4), ('MSD_15to1_validation', vmsd)]:
    if v is None:
        rows.append(dict(notebook=name, checks=None, PASS=None, FAIL=None, not_verified='（未取得：validation ノートの結果ファイルが見つからない）'))
        continue
    res = v['results']
    rows.append(dict(notebook=v['notebook'], checks=len(res), PASS=sum(r['status'] == 'PASS' for r in res.values()),
                     FAIL=sum(r['status'] == 'FAIL' for r in res.values()), not_verified='; '.join(v['not_verified'])))
val_df = pd.DataFrame(rows); val_df'''),
    md(r'''## 1. 段階別の資源コスト一覧（$d=9$、`FT_ref`、比較A、データ＋測定量子ビット、原始ゲートセット）'''),
    code(r'''d_show = 9 if 9 in D_LIST else D_LIST[len(D_LIST) // 2]
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
tab'''),
    md(r'''`rounds` は Kobayashi では FSC／D4 のラウンド、Litinski では回転表面符号のラウンド。単位が違うので、横並びにする量は物理時間（`time_lb/ub`、全ゲート1単位の仮定）である。

## 2. 主比較：$d$ 依存性（`FT_ref`、比較A、データ＋測定量子ビット）'''),
    code(r'''def pick(df, **kw):
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
main.round(3)'''),
    md(r'''* `ratio_V_time_lb` は両方式の下界同士、`ratio_V_time_ub` は上界同士の比。`ratio_V_time_range` は最も保守的な範囲（K下界/L上界 〜 K上界/L下界）。
* Kobayashi の値は $\kappa_0=\kappa_{in}=\kappa_{out}=1$（**仮定**）に依存する。下の §4 で $\kappa$ 依存性を示す。'''),
    code(r'''ds = main.d.values
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
plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_main_FT_ref_A.png', dpi=150); plt.show()'''),
    md(r'''## 3. 先頭係数（漸近的スケーリング）
最初の 4 点で補間した多項式が全ての点で厳密に一致するかを確認する（$\kappa=1$、`FT_ref`、比較A、データ＋測定量子ビット）。'''),
    code(r'''rows = []
sel = {('Kobayashi', 'FT_ref'): K, ('Litinski physical', 'FT_ref'): Lp, ('Litinski tile', 'FT_ref'): Lt}
for (name, model), df in sel.items():
    for q in ['peak_qubits', 'rounds', 'V_qubit_round', 'V_qubit_time_lb', 'V_qubit_time_ub']:
        p, ok = identify_polynomial(dict(zip(df.d, df[q])))
        rows.append(dict(protocol=name, quantity=q, polynomial_in_d=str(p), exact_on_all_points=ok))
poly_df = pd.DataFrame(rows); poly_df.to_csv(f'{RESULTS}/comparison_polynomials.csv', index=False); poly_df'''),
    md(r'''まとめると（データ＋測定量子ビット、`FT_ref`、比較A）：
* Kobayashi：$V_{\rm qubit\cdot round} = (8\kappa_0+12\kappa_{in}+8\kappa_{out})d^3 + O(d^2)$ → $\kappa=1$ で $28d^3+\dots$
* Litinski：$V_{\rm qubit\cdot round} = 242d^3 - 66d$（物理モデル＝明示的構成）、$242d^3$（tile モデル、測定量子ビット込み）、$121d^3$（tile、データのみ＝[Lit] の値）
* ピーク量子ビット：Kobayashi $12d^2+4d-1$、Litinski $22d^2-5$（物理モデル）

ただし Kobayashi の 1 ラウンド（D4 相）は物理時間で回転表面符号の 1 ラウンドより長い（上界 28 vs 12、下界 9 vs 6、全ゲート1単位）。物理時間で測った体積の比は §2 の表を参照。

## 4. $\kappa$（$O(d)$ の定数）への依存性と損益分岐
[KZH] は switch-in／EC のラウンド数を $O(d)$ としか述べていないため、$\kappa$ を振る。`d4_blocks.json` の構成要素から体積を再構成する（$\kappa=1$ で CSV と一致することを確認）。'''),
    code(r'''import math
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
be_df = pd.DataFrame(rows); be_df.to_csv(f'{RESULTS}/comparison_breakeven_kappa.csv', index=False); be_df'''),
    md(r"""## 4b. ゲート所要時間（CCZ）への感度
§2 の物理時間は全ゲート 1 単位という仮定に依存する。D4 側のラウンドは CCZ を含むので、CCZ の所要時間 $t_{\rm CCZ}$ を 1〜12 単位で変え、$\kappa_{in}$ と合わせて体積比（物理時間、上界同士・下界同士）がどう変わるかを示す（Litinski 側は CCZ を使わないので不変）。"""),
    code(r"""t_list = [1, 2, 3, 5, 8, 12]
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
sens_df.round(3)"""),
    md(r"""CCZ が遅いほど D4 のラウンドが長くなり、損益分岐の $\kappa_{in}$ は小さくなる。比 1 の等高線より左下が「Kobayashi 方式の体積が小さい」領域である（他の仮定はすべて既定値）。

"""),
    md(r'''損益分岐 $\kappa_{in}$：これより大きい定数で switch-in に $\kappa_{in}d$ ラウンドが必要なら、Kobayashi 方式の体積が Litinski 方式（比較A、物理モデル）を上回る。主要項だけで見ると $V_{\rm qubit\cdot round}$ では $8\kappa_0+12\kappa_{in}+8\kappa_{out}=242$ より $\kappa_{in}\approx 18.8$（$\kappa_0=\kappa_{out}=1$）。物理時間の体積では D4 ラウンドの方が深いので損益分岐はより小さい（表）。

## 5. ノイズなし最小モデル（`NL_min`）
繰り返し測定を一切しない論理的最小。**耐故障性の主張ではない**（§3.3）。'''),
    code(r'''Kn = pick(d4s, model='NL_min', comparison='A', gateset='primitive', data_only=False)
Ln = pick(msds, model='NL_min', comparison='A', gateset='primitive', data_only=False, level='physical')
nl = pd.DataFrame(dict(d=Kn.d.values, K_peak=Kn.peak_qubits.values, L_peak=Ln.peak_qubits.values,
                       K_rounds=Kn.rounds.values, L_rounds=Ln.rounds.values, K_time_lb=Kn.time_lb.values, K_time_ub=Kn.time_ub.values,
                       L_time_lb=Ln.time_lb.values, L_time_ub=Ln.time_ub.values,
                       K_V_qr=Kn.V_qubit_round.values, L_V_qr=Ln.V_qubit_round.values,
                       ratio_V_time_lb=Kn.V_qubit_time_lb.values / Ln.V_qubit_time_lb.values,
                       ratio_V_time_ub=Kn.V_qubit_time_ub.values / Ln.V_qubit_time_ub.values))
nl.to_csv(f'{RESULTS}/comparison_NL_min_A.csv', index=False); nl.round(3)'''),
    md(r'''## 6. 比較B：可能な範囲での完全な資源比較

追加で計上するもの
* Litinski：$|m\rangle$ 注入 15 回（各 $t_{\rm inj}=6$ cycles、[Lit] App. A の記述；11-tile 配置では回転ごとに直列）、物理 $T$ ゲート 15 個。
* Kobayashi：物理非 Clifford ゲート（CCZ：測定ガジェット、CS/CS$^\dagger$/$T^\dagger$：$U$）。Clifford+T 分解時の T 数と、分解でラウンドが深くなる効果。
* **FSC → 通常の回転表面符号パッチ変換：未解決**。これは**合計に 0 として含めず**、下の「未計上コスト」の表に分けて示す。したがって比較Bの Kobayashi の合計は変換を含まない**下界**である。参考として「1τ の格子手術（FSC＋2 tiles）」という**例示的仮定**を加えた値も別列に示す（検証された手順ではない）。'''),
    code(r'''unc = pd.DataFrame([dict(protocol=p, comparison=c, uncounted_cost=a, status=b)
                    for p in ['Kobayashi_D4', 'Litinski_15to1'] for c in ['A', 'B'] for a, b in uncounted_costs(p, c)])
unc.to_csv(f'{RESULTS}/uncounted_costs.csv', index=False); unc'''),
    code(r'''rows = []
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
Bdf[Bdf.gateset == 'primitive'][['d', 'K_peak', 'L_peak', 'K_V_qr', 'K_V_qr_with_illustrative_conversion', 'L_V_qr', 'K_nonclifford', 'L_nonclifford', 'K_Tcount_if_decomposed', 'L_Tcount']]'''),
    code(r'''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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
plt.tight_layout(); plt.savefig(f'{FIGURES}/comparison_B.png', dpi=150); plt.show()'''),
    md(r'''## 7. 仮定・未確定事項の一覧（表）
詳細な説明は `docs/02_assumptions_and_open_issues.md`。'''),
    code(r'''assump = pd.DataFrame([
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
assump.to_csv(f'{RESULTS}/assumptions_table.csv', index=False); assump'''),
    md(r'''## 8. 結論（理想条件下、指定した 2 プロトコルの回路構造に関する**予備的**評価）

1. **正しさ**：Kobayashi 方式は、境界付き D4 符号（Fig. 4）上で $U=WV$ が論理 $T^\dagger$ を実装し、ランダムな測定結果に対する補正（$Z^rZ^b$ の Pauli フレーム補正と $S^g_X$ の CZ 補正）を含めて FSC 上に $T^\dagger|+\rangle$ を生成する。根拠は、$L\le2$ の符号状態の厳密シミュレーション、$L\le4$ での回路と位相公式の照合、$L\le 8$ の位相公式の局所厳密証明（それぞれ意味の異なる検証）。D4 相の距離が $L$ であることは $L\le5$（6）で証明済み。Litinski 方式は 15-to-1 の論理動作・誤り検出・auto-corrected π/8・11 tiles × 11τ・$121d^3$ を再現し、11 回の多パッチ測定を物理レベルで明示的に構成・検証した（$Z_m\otimes Y_0$ の twist 継ぎ目を除く）。
2. **量子ビット数**（計数）：ピークで Kobayashi $12L^2+4L-1$ に対し Litinski $22d^2-6$（データ＋測定量子ビット、$L=d$）。
3. **時間・体積**（仮定に依存）：$\kappa=1$・全ゲート 1 単位では、Kobayashi は $3d$ ラウンド＋定数深さ、Litinski は $11d$ ラウンドで、体積比は約 0.1〜0.2。しかしこれは仮定に強く依存する：損益分岐 $\kappa_{in}$ は qubit·round 体積で約 15〜19、物理時間体積で約 6〜13（§4）、さらに CCZ が遅いと小さくなる（§4b）。**$O(d)$ の定数とゲート時間が確定するまで、優劣は確定しない**。
4. **比較B**：Kobayashi 方式は $\Theta(d^3)$ 個の物理 CCZ（Clifford+T 分解で $\approx 42d^3$ 個の T）を使うのに対し、Litinski 方式の物理非 Clifford 資源は T 15 個。FSC→通常パッチ変換は未解決で、合計には含めず「未計上コスト」として分離した。比較Bは**共通のスタートとゴールからの完全な比較としては未完成**である。
5. **限界**：ノイズ0では蒸留は本来不要であり、本比較は最適な T ゲート実装法を決めるものではない。論文の主張として使うには、ノイズ下での $\kappa$ の決定、変換回路の構成、ハードウェアに即したゲート時間が必要である。
'''),
]
save('resource_comparison.ipynb', cells)
