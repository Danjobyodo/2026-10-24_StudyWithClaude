"""Builds noisy_cost_comparison.ipynb: cost comparison of D4 code switching vs 15-to-1 distillation under noise."""
import os
from nbutil import *

ROOT = OUT


def _embed(path):
    p = os.path.join(ROOT, path)
    return open(p).read() if os.path.exists(p) else ''


EMBED_MC = _embed('results/d4_noisy_mc.csv')
EMBED_CAL = _embed('results/surface_code_plog.csv')
CONCLUSION = r'''
## 結論
@@CONCLUSION@@
'''

cells = [md(r'''
# noisy_cost_comparison — 物理誤り率を与えたときの、Kobayashi 方式（D4 code switching）と Litinski 方式（15-to-1 蒸留）の資源比較

**問い**：物理誤り率 $p$ と要求される出力誤り $\varepsilon_{\rm target}$ が与えられたとき、論理 $T$ 魔法状態 1 個を作るのに、どちらの方式が少ない時空間体積（物理量子ビット × 時間ステップ）で済むか。

**このノートがすること**（重い計算はしない。数分で終わる）
1. **Kobayashi 方式**：`D4_noisy_simulation.ipynb` の回路レベル・モンテカルロの結果 $\varepsilon_K(d,p)$（非 Clifford 回路を Takada らの方法で厳密にシミュレーション）を読み込み、$d$ と $p$ への依存を当てはめる。資源は同じ回路から数える。
2. **Litinski 方式**：Litinski（2019）の公開コードの誤り予算（本研究で再実装し、公開値 7 件を再現：`validation.ipynb` N6）を、*A Game of Surface Codes*（GoSC）Sec. 3 の 11-tile 15-to-1 と 176-tile 225-to-1、および Litinski（2019）の最適化された蒸留器に適用する。パッチの論理誤り率 $p_L(d,p)$ は**同じ雑音モデル**の表面符号シミュレーションで較正する。
3. 同じ単位（物理量子ビット × 時間ステップ、全ゲート 1 時間ステップ）で比較する。

**雑音のシナリオ**：A＝アイドル雑音あり（$p_{\rm idle}=p$）、B＝アイドル雑音なし（Takada らと同じ）。ゲート・初期化・測定の誤りはどちらも $p$。

**結果の位置づけ**：Kobayashi 方式の数値は**本研究で構成した復号器**（[KZH] が引用する JIT decoder [Dav] は参照できなかった）と**最適化していない測定スケジュール**に依存し、最適な復号器・スケジュールで達成できる値の上界（保守的）である。Litinski 方式は解析的な誤り予算モデル（文献の手法）である。手法・仮定の詳細は `docs/methods.md`。
'''),
    code(SETUP), code(INSTALL), code(PLOTSTYLE),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'noisy_d4_lib.py', 'msd_noisy_lib.py'),
    md(r'''
## §0. パラメータとデータ
* `P_EVAL`：比較する物理誤り率。`TARGETS`：要求出力誤り。
* `STEPS_PER_CYCLE`：表面符号 1 ラウンド（code cycle）の時間ステップ数。較正に使った回路（初期化 → CNOT×4 → 測定）と同じ 6（`validation.ipynb` N7）。
* `PLOG_SOURCE`：`'calibrated'`（同じ雑音モデルでの表面符号シミュレーション、既定）または `'gosc'`（GoSC 式 (10)：$0.1(100p)^{(d+1)/2}$）。
* モンテカルロの結果ファイルが無い場合（Colab で単独実行した場合など）は、このノートに埋め込まれた本番結果を使う。
'''),
    code(r'''
P_EVAL = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
TARGETS = [1e-4, 1e-6, 1e-8, 1e-10]
SCENARIOS = {'A': 'アイドル雑音あり (p_idle = p)', 'B': 'アイドル雑音なし (p_idle = 0)'}
STEPS_PER_CYCLE = 6
PLOG_SOURCE = 'calibrated'
import io
EMBED_MC = """''' + EMBED_MC + r'''"""
EMBED_CAL = """''' + EMBED_CAL + r'''"""
def _load(path, embedded):
    if os.path.exists(path):
        print('read', path); return pd.read_csv(path)
    print(path, 'not found -> embedded copy (production run)'); return pd.read_csv(io.StringIO(embedded))
mc = _load(f'{RESULTS}/d4_noisy_mc.csv', EMBED_MC)
cal = _load(f'{RESULTS}/surface_code_plog.csv', EMBED_CAL)
print(len(mc), 'Monte Carlo chunks,', len(cal), 'calibration points')
'''),
    md(r'''
## 第I部 表面符号の論理誤り率の較正（Litinski 方式の入力）
1 ラウンドあたりの論理誤り（X と Z の和）$p_L$ を $p_L = A\,(p/p_{\rm th})^{(d+1)/2}$ に当てはめる（GoSC 式 (10) と同じ形。GoSC は $A=0.1$, $p_{\rm th}=0.01$）。閾値より十分下の $p\le0.004$ で、失敗が 20 回以上ある点を使う。
'''),
    code(r'''
PLOG = {}
rows = []
for s in SCENARIOS:
    c_ = cal[(cal.scenario == s) & (cal.pL_per_round > 0)]
    w_ok = c_[((c_.PL_x + c_.PL_z) * c_.shots >= 20) & (c_.p <= 0.004)]   # enough failures, below threshold
    A, pth = fit_plog(list(zip(w_ok.d, w_ok.p, w_ok.pL_per_round)))
    PLOG[s] = make_plog(A, pth) if PLOG_SOURCE == 'calibrated' else plog_gosc
    rows.append(dict(scenario=s, A=A, p_th=pth, points=len(w_ok),
                     ratio_to_GoSC_at_p1e3_d11=float(make_plog(A, pth)(1e-3, 11) / plog_gosc(1e-3, 11))))
pd.DataFrame(rows).round(5)
'''),
    md(r'''
## 第II部 Kobayashi 方式
### II-1. モンテカルロの結果
ショットごとの不忠実度（0, 1/2, 1）の平均 $\hat\varepsilon_K$ と標準誤差。$\kappa_{\rm in}\neq1$ の行は switch-in のラウンド数の感度。
'''),
    code(r'''
g = mc.groupby(['scenario', 'd', 'p', 'kappa_in']).agg(shots=('shots', 'sum'), s=('sum_infidelity', 'sum'),
                                                       n_fail=('n_fail', 'sum'), n_half=('n_half', 'sum'),
                                                       errors_per_shot=('mean_errors', 'mean')).reset_index()
g['eps'] = g.s / g.shots
n_one = g.n_fail - g.n_half
g['stderr'] = np.sqrt(np.maximum((n_one + 0.25 * g.n_half) / g.shots - g.eps ** 2, 0) / g.shots)
main = g[g.kappa_in == 1.0].copy()
g.round({'eps': 6, 'stderr': 6, 'errors_per_shot': 1})
'''),
    md(r'''
### II-2. $d$ と $p$ への依存の当てはめ（外挿の根拠）
閾値以下では $\varepsilon_K \approx A\,\Lambda(p)^{-d}$、$\Lambda$ は $p$ のべきで決まる、という標準的な形を仮定し、
$$\log\varepsilon_K = a + c\,d + (b + e\,d)\log p$$
を、失敗が 10 回以上ある点に（標準誤差で重み付けして）当てはめる。$d$ 方向の抑制が消える $p$（$c + e\log p = 0$）を擬閾値 $p^*$ とする。**$d\le 9$ のデータからの外挿であり、確定値ではない**。
'''),
    code(r'''
FIT = {}
rows = []
for s in SCENARIOS:
    m_ = main[(main.scenario == s) & (main.n_fail >= 10) & (main.eps > 0)]
    X_ = np.stack([np.ones(len(m_)), m_.d, np.log(m_.p), m_.d * np.log(m_.p)], axis=1)
    y_ = np.log(m_.eps.values)
    w_ = (m_.eps / np.maximum(m_.stderr, 1e-12)).values           # 1/sigma of log(eps)
    coef, *_ = np.linalg.lstsq(X_ * w_[:, None], y_ * w_, rcond=None)
    FIT[s] = coef
    a, c, b, e = coef
    pstar = float(np.exp(-c / e)) if e != 0 else float('nan')
    resid = (X_ @ coef - y_)
    rows.append(dict(scenario=s, a=a, c=c, b=b, e=e, pseudo_threshold=pstar, points=len(m_),
                     rms_log_residual=float(np.sqrt(np.mean(resid ** 2)))))
def eps_K_model(s, d, p):
    a, c, b, e = FIT[s]
    return float(np.exp(a + c * d + (b + e * d) * np.log(p)))
pd.DataFrame(rows).round(4)
'''),
    code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
mk = {3: 'o', 5: 's', 7: '^', 9: 'D'}
cols = {3: '#8fb8ea', 5: '#5a95dc', 7: '#2a78d6', 9: '#174a8a'}
for ax, s in zip(axes, SCENARIOS):
    for d in sorted(main.d.unique()):
        m_ = main[(main.scenario == s) & (main.d == d)].sort_values('p')
        ax.errorbar(m_.p, m_.eps, yerr=m_.stderr, fmt=mk.get(d, 'o'), color=cols.get(d, 'k'), label=f'd = {d} (simulated)', capsize=2)
        pp = np.logspace(-4.2, -2.9, 40)
        ax.plot(pp, [eps_K_model(s, d, x) for x in pp], '-', color=cols.get(d, 'k'), lw=1, alpha=0.7)
    pp = np.logspace(-4.2, -2.9, 40)
    ax.plot(pp, pp, ':', color='#52514e', lw=1, label='raw injected state (error ~ p)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('physical error rate p')
    ax.set_title(f'Kobayashi D4 switching, scenario {s}')
axes[0].set_ylabel('output infidelity  ε_K(d, p)')
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_logical_error.png', dpi=150); plt.show()
'''),
    md(r'''
### II-3. 資源（同じ回路から数える）
量子ビット数（ピーク）と時間ステップ数を、シミュレーションに使ったのと同じ回路（$\kappa_0=\kappa_{\rm in}=\kappa_{\rm out}=1$）から $d=3,\dots,11$ で数え、$d$ の多項式に当てはめて大きな $d$ に延長する。体積は「各ブロックで使う量子ビット数 × そのブロックの時間ステップ数」の和（雑音なしの最終ラウンドは含めない）。
**含めていない費用**：FSC から通常の回転表面符号パッチへの変換（未構成、[KZH] にも記述なし）、古典処理の待ち時間。
'''),
    code(r'''
def kobayashi_counts(d):
    circ = D4SwitchingCircuit(d, d, d, d)
    comp = CompiledProtocol(circ, NoiseModel(1.0))
    vol, steps = 0, 0
    for blk in comp.blocks:
        if blk['name'][0] == 'FIN':
            continue
        code_ = blk['code']
        active = set(circ.data_rb if (code_ == 'FSC') else circ.data_all) | (set(circ.anc[code_]) if code_ in ('FSC',) else set())
        if code_ == 'D4':
            active |= set(circ.anc['D4'])
        n_steps = len(blk['moms'])
        vol += len(active) * n_steps
        steps += n_steps
    return dict(d=d, qubits=circ.nq, time_steps=steps, volume=vol)
KC = pd.DataFrame([kobayashi_counts(d) for d in (3, 5, 7, 9, 11)])
POLY = {k: np.polyfit(KC.d, KC[k], 3 if k == 'volume' else 2) for k in ('qubits', 'time_steps', 'volume')}
def kobayashi_cost(d):
    return {k: float(np.polyval(POLY[k], d)) for k in POLY}
KC['volume_poly'] = [kobayashi_cost(d)['volume'] for d in KC.d]
print({k: np.round(v, 2) for k, v in POLY.items()})
KC
'''),
    md(r'''
## 第III部 Litinski 方式（誤り予算モデル）
* **GoSC 15-to-1**（Fig. 17c：11 tiles × 11τ、1τ = $d$ cycles）：単一の距離 $d$。
* **GoSC 225-to-1**（Fig. 19：176 tiles × 15τ）：1 段目 $d_1$、2 段目 $d_2$。
* **Litinski 2019 15-to-1** $(d_X,d_Z,d_m)$ と **(15-to-1)×(15-to-1)**：Litinski（2019）の配置と費用式（公開コードと同じ）。
* 入力の魔法状態：確率 $p$ の Pauli 誤り（GoSC Sec. 4 の仮定）。出力誤りは事後選択後の不忠実度、時間は成功確率で割る。
* 量子ビット数はデータ＋測定量子ビット（1 tile = $2d^2$）、時間は code cycle × `STEPS_PER_CYCLE`。
'''),
    code(r'''
import itertools as _it
def litinski_candidates(p, s):
    pl = PLOG[s]
    out = []
    for d in range(3, 52, 2):
        out.append(gosc_15to1(p, d, pl))
    lvl1 = {d1: run_steps(gosc_15to1_steps(p, d1, pl), out_qubit=4) for d1 in range(3, 26, 2)}
    for d1 in range(3, 26, 2):
        for d2 in range(d1, 52, 2):
            out.append(gosc_225to1(p, d1, d2, pl, level1=lvl1[d1]))
    for dx in range(5, 40, 2):
        for dz in range(3, dx + 1, 2):
            out.append(litinski2019_one_level(p, dx, dz, dz, pl))
    for dx, dz, dx2, dz2, nl1 in _it.product((7, 9, 11, 13), (3, 5), range(13, 42, 2), (5, 7, 9, 11, 13), (4, 6)):
        if dz < dx and dz2 < dx2:
            out.append(litinski2019_two_level(p, dx, dz, dz, dx2, dz2, dz2, nl1, pl))
    df = pd.DataFrame(out)
    df['family'] = df.protocol.str.extract(r'^(GoSC 15-to-1|GoSC 225-to-1|Litinski2019 \(15-to-1\)x|Litinski2019 \(15-to-1\))')[0]
    df['time_steps'] = df.cycles * STEPS_PER_CYCLE
    df['volume_steps'] = df.qubits * df.time_steps
    return df
import time as _time
t0 = _time.time()
LIT = {(s, p): litinski_candidates(p, s) for s in SCENARIOS for p in P_EVAL}
print('evaluated', sum(len(v) for v in LIT.values()), 'factory configurations in %.0f s' % (_time.time() - t0))
'''),
    md(r'''
## 第IV部 比較
### IV-1. 出力誤り vs 時空間体積
各方式で、ある体積で到達できる最小の出力誤り（パレート前線）。Kobayashi 方式は $d=3,\dots,9$ がシミュレーション（実線・マーカー）、それより大きい $d$ は II-2 の当てはめによる外挿（破線）。
'''),
    code(r'''
def pareto(df):
    df = df.sort_values('volume_steps')
    best, rows = np.inf, []
    for _, r in df.iterrows():
        if r.eps < best:
            best = r.eps; rows.append(r)
    return pd.DataFrame(rows)
show_p = [1e-4, 3e-4, 1e-3]
fig, axes = plt.subplots(2, len(show_p), figsize=(4.2 * len(show_p), 7.6), sharex=True, sharey=True)
for i, s in enumerate(SCENARIOS):
    for j, p in enumerate(show_p):
        ax = axes[i, j]
        fr = pareto(LIT[(s, p)])
        ax.step(fr.volume_steps, fr.eps, where='post', color=C_LIT, label='Litinski (best of GoSC / 2019 factories)')
        ds = np.arange(3, 61, 2)
        ek = np.array([eps_K_model(s, d, p) for d in ds]); vk = np.array([kobayashi_cost(d)['volume'] for d in ds])
        ax.plot(vk, np.minimum(ek, 1), '--', color=C_KOB, lw=1.2, label='Kobayashi (fit / extrapolation)')
        sim = main[(main.scenario == s) & (main.p == p)]
        ax.errorbar([kobayashi_cost(d)['volume'] for d in sim.d], sim.eps, yerr=sim.stderr, fmt='o', color=C_KOB,
                    label='Kobayashi (simulated d = 3..9)')
        ax.axhline(p, color='#52514e', ls=':', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-14, 1)
        ax.set_title(f'scenario {s},  p = {p:g}')
        if j == 0: ax.set_ylabel('output error of one T state')
        if i == 1: ax.set_xlabel('space-time volume [physical qubits x time steps]')
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIGURES}/cost_comparison.png', dpi=150); plt.show()
'''),
    md(r'''
### IV-2. 要求誤りごとの最小費用
`Kobayashi d` が 9 を超える場合は外挿、`—` は当てはめの範囲で到達できない（擬閾値より上、または $d\le 61$ で届かない）ことを示す。
'''),
    code(r'''
rows = []
for s in SCENARIOS:
    for p in P_EVAL:
        df = LIT[(s, p)]
        for t in TARGETS:
            ok = df[df.eps <= t]
            lit = ok.loc[ok.volume_steps.idxmin()] if len(ok) else None
            dk = next((d for d in range(3, 62, 2) if eps_K_model(s, d, p) <= t), None)
            kc = kobayashi_cost(dk) if dk else None
            rows.append(dict(scenario=s, p=p, target=t,
                             litinski_protocol=lit.protocol if lit is not None else '—',
                             litinski_eps=lit.eps if lit is not None else np.nan,
                             litinski_qubits=lit.qubits if lit is not None else np.nan,
                             litinski_time_steps=lit.time_steps if lit is not None else np.nan,
                             litinski_volume=lit.volume_steps if lit is not None else np.nan,
                             kobayashi_d=dk if dk else '—',
                             kobayashi_extrapolated=(dk is not None and dk > 9),
                             kobayashi_eps_model=eps_K_model(s, dk, p) if dk else np.nan,
                             kobayashi_qubits=kc['qubits'] if kc else np.nan,
                             kobayashi_time_steps=kc['time_steps'] if kc else np.nan,
                             kobayashi_volume=kc['volume'] if kc else np.nan))
CMP = pd.DataFrame(rows)
CMP['volume_ratio_K_over_L'] = CMP.kobayashi_volume / CMP.litinski_volume
CMP.to_csv(f'{RESULTS}/cost_comparison.csv', index=False)
pd.set_option('display.max_colwidth', 60)
CMP[['scenario', 'p', 'target', 'litinski_protocol', 'litinski_volume', 'kobayashi_d', 'kobayashi_extrapolated',
     'kobayashi_volume', 'volume_ratio_K_over_L']]
'''),
    md(r'''
### IV-3. Kobayashi 方式が同じ体積に収まるために必要な改善
要求誤りごとに、Litinski 方式の最小体積以下に収まる Kobayashi 方式の最大の $d$（$d_{\max}$）を求め、そこでの $\varepsilon_K(d_{\max},p)$（当てはめ）が要求誤りの何倍か（`required_improvement`）を示す。復号器・スケジュールの改良でこの倍率だけ $\varepsilon_K$ が下がれば、体積で並ぶ（FSC → パッチ変換の費用は含めていない）。
'''),
    code(r'''
rows = []
for _, r in CMP.iterrows():
    if not np.isfinite(r.litinski_volume):
        continue
    ds = [d for d in range(3, 62, 2) if kobayashi_cost(d)['volume'] <= r.litinski_volume]
    dmax = max(ds) if ds else None
    ek = eps_K_model(r.scenario, dmax, r.p) if dmax else np.nan
    rows.append(dict(scenario=r.scenario, p=r.p, target=r.target, litinski_volume=r.litinski_volume, kobayashi_d_max=dmax,
                     eps_K_at_d_max=ek, required_improvement=ek / r.target if dmax else np.nan))
IMP = pd.DataFrame(rows)
CMP = CMP.merge(IMP[['scenario', 'p', 'target', 'kobayashi_d_max', 'required_improvement']], how='left',
                on=['scenario', 'p', 'target'])
CMP.to_csv(f'{RESULTS}/cost_comparison.csv', index=False)
IMP.round({'eps_K_at_d_max': 8, 'required_improvement': 1})
'''),
    md(CONCLUSION),
]
save('noisy_cost_comparison.ipynb', cells)
