"""Builds noisy_cost_comparison.ipynb: cost comparison of D4 code switching vs 15-to-1 distillation under noise."""
import os
import hashlib
from nbutil import *

ROOT = OUT


def _embed(path):
    p = os.path.join(ROOT, path)
    return open(p).read() if os.path.exists(p) else ''


EMBED_MC = _embed('results/d4_noisy_mc.csv')
EMBED_CAL = _embed('results/surface_code_plog.csv')
_sha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
CONCLUSION = r'''
## 結論
本研究で構成した回路・スケジュール・復号器と、§II-2 のモデル（外挿を含む）に基づく結論。データ：`production-20261029T053925Z-29f79c`（329 万ショット）。数値は `results/cost_comparison.csv`。

1. **本来のシミュレーション**：Kobayashi 方式の出力誤りは $d\ge5$ で $d$ をほとんど下げず、シナリオ B で $\varepsilon_K\approx(3\text{–}8)\,p$、A ではさらに大きい（$p=10^{-3}$ で約 20%）。主因は切り替えの窓での単一故障で、`validation.ipynb` N5 で窓を全列挙した一次の係数は $c_1=3.1$（B, $d=5$）、$3.9$（B, $d=7$）、$3.0$（A, $d=5$）と $d\ge5$ でほぼ一定、測定値と整合する。このモデル（窓の項の $d$ 依存を許した感度モデルでも）では、評価した全ての $(p,$ 要求誤り$)$ で要求に届かない（`not_reached_model`）。これは**本研究の回路・復号器での結果**であり、方式として不可能という意味ではない。
2. **窓の雑音を除いた診断**（同じ復号器）では $\varepsilon_K$ は $d$ とともに下がる（測定範囲の端 $p=3\times10^{-4}$ で、当てはめたモデルの減少率は距離 2 ごとに A 約 4.2 倍、B 約 3.5 倍）。この診断のモデルを費用比較に入れると、要求 $10^{-6}$ 以下では Kobayashi 方式の体積は Litinski 方式（候補集合内の最小）の 2.7〜39 倍（A：4.2〜39 倍、$p=10^{-3}$ では届かない；B：2.7〜33 倍）。並ぶか下回るのは要求 $10^{-4}$ の一部（B・$p=10^{-4}$ で 0.52 倍）だけ。多くの行は $d>9$ や低い $p$ への外挿で、ブートストラップの範囲は表の `_16`/`_84` 列。
3. **解釈**：本研究の実装では、Kobayashi 方式の性能は切り替えの窓の扱いで決まっている。窓の雑音を除いた診断でも蒸留の方が体積が小さい傾向が残るが、診断は最適な復号器の性能の限界（上界・下界）ではなく、また外挿に依存するので、**最適化された D4 方式に対する一般的な優劣を示すものではない**。
4. 費用の差の内訳：D4 相の 1 ラウンドが長く（CCZ を含む X 型安定化子の測定 28 ステップ＋$Z$ 検査 6 ステップ）雑音の位置が多い、出力 1 個のために $d\times d$ 格子全体を $O(d)$ ラウンド使う（体積 $\approx632d^3$）。switch-in のラウンド数は $\kappa_{\rm in}\ge0.6$ で $\varepsilon_K$ がほぼ変わらない（窓の項が支配するため）。
5. **結論を変えうる要因**（いずれも未解決）：切り替えの窓を扱える復号器（[Dav] の JIT decoder）や回路の改良、測定スケジュールの最適化（FSC 14 → 6 ステップ）、CCZ の実際の誤り率、FSC → 通常パッチの変換（未計上）、Litinski 側の状態注入（未計上）、Litinski 側を回路レベルでシミュレーションした場合との差。
'''

LIT_CODE = r"""
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
"""

cells = [md(r'''
# noisy_cost_comparison — 物理誤り率を与えたときの、Kobayashi 方式（D4 code switching）と Litinski 方式（15-to-1 蒸留）の資源比較

**問い**：物理誤り率 $p$ と要求される出力誤り $\varepsilon_{\rm target}$ が与えられたとき、論理 $T$ 魔法状態 1 個を作るのに、どちらの方式が少ない時空間体積（物理量子ビット × 時間ステップ）で済むか。

**このノートがすること**（重い計算はしない。数分で終わる）
1. **Kobayashi 方式**：`D4_noisy_simulation.ipynb` の回路レベル・モンテカルロの結果 $\varepsilon_K(d,p)$（非 Clifford 回路を Takada らの方法でシミュレーション）を読み込み、$d$ と $p$ への依存のモデルを当てはめる（不確かさはブートストラップ）。資源は同じ回路から数える。
2. **Litinski 方式**：Litinski（2019）の公開コードの誤り予算（本研究で再実装し、公開値 8 件を再現：`validation.ipynb` N6）を、*A Game of Surface Codes*（GoSC）Sec. 3 の 11-tile 15-to-1 と 176-tile 225-to-1、および Litinski（2019）の最適化された蒸留器に適用する。パッチの論理誤り率 $p_L(d,p)$ は同じ雑音パラメータの表面符号シミュレーションで較正する。
3. 同じ単位（物理量子ビット × 時間ステップ、全ゲート 1 時間ステップ）で比較する。

**比較の非対称性**：Kobayashi 側は切り替えを含む回路全体のモンテカルロ、Litinski 側は「較正した表面符号メモリの誤り率」を入れた解析的な誤り予算モデル（格子手術や蒸留器全体を回路レベルでシミュレーションしたものではない）。どちらにも計上していない費用がある（Kobayashi：FSC → 通常パッチの変換、Litinski：状態注入、両方：古典処理の待ち時間）。比較は「各方式の符号の中で 1 個の $T$ 状態を作る費用」であり、共通の出力パッチへ受け渡す費用までは含まない。

**雑音のシナリオ**：A＝アイドル雑音あり（$p_{\rm idle}=p$）、B＝アイドル雑音なし（Takada らと同じ）。ゲート・初期化・測定の誤りはどちらも $p$。

**結果の位置づけ**：Kobayashi 方式の数値は**本研究で構成した回路・測定スケジュール・復号器**（[KZH] が引用する JIT decoder [Dav] は参照できなかった）での値であり、Kobayashi 方式一般の性能ではない。手法・仮定の詳細は `docs/methods.md`。
'''),
    code(SETUP), code(INSTALL), code(PLOTSTYLE),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'noisy_d4_lib.py', 'msd_noisy_lib.py'),
    md(r'''
## §0. パラメータと入力データ（データの来歴）
* `DATA_MODE`：解析するデータを**明示的に**選ぶ。自動の切り替え（fallback）はしない。
  * `'reference'`（既定）：このノートに埋め込まれた本番データ（リポジトリの `results/` と同一。下の SHA-256 で照合できる）。リポジトリの比較表・図を再現する。
  * `'production'`：`D4_noisy_simulation.ipynb` を `QUICK = False` で実行して得た `results/d4_noisy_mc.csv`, `results/surface_code_plog.csv`。無ければ停止する。
  * `'quick'`：`QUICK = True` の動作確認の結果 `results/*_quick.csv`。無ければ停止する。点数が少ないのでモデルを当てはめられず、§II-2 で停止する（集計の確認まで）。
* Colab ではノートごとに別のランタイムなので、`production` / `quick` を使う場合はシミュレーションのノートで作られた `results/` の 2 つの CSV を、このノートのランタイムの `results/` にアップロードする（または両方のノートで同じ Google Drive のフォルダを作業ディレクトリにする）。
* 入力の SHA-256（先頭 16 桁）、実行 ID（`run_id`）、ショット数を表示し、比較表（`cost_comparison.csv`）の各行にも記録する。
* `P_EVAL`：比較する物理誤り率。`TARGETS`：要求出力誤り。`N_BOOT`：ブートストラップの回数。
* `STEPS_PER_CYCLE`：表面符号 1 ラウンド（code cycle）の時間ステップ数（較正に使った回路と同じ 6：`validation.ipynb` N7）。
* `PLOG_SOURCE`：`'calibrated'`（較正、既定）または `'gosc'`（GoSC 式 (10)：$0.1(100p)^{(d+1)/2}$）。
'''),
    code(r'''
DATA_MODE = 'reference'
P_EVAL = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
TARGETS = [1e-4, 1e-6, 1e-8, 1e-10]
N_BOOT = 200
SCENARIOS = {'A': 'アイドル雑音あり (p_idle = p)', 'B': 'アイドル雑音なし (p_idle = 0)'}
STEPS_PER_CYCLE = 6
PLOG_SOURCE = 'calibrated'
import io, hashlib, platform
EMBED_MC = """''' + EMBED_MC + r'''"""
EMBED_CAL = """''' + EMBED_CAL + r'''"""
REF_SHA = {'mc': ''' + repr(_sha(EMBED_MC)) + r''', 'cal': ''' + repr(_sha(EMBED_CAL)) + r'''}
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
if DATA_MODE == 'reference':
    mc_text, cal_text, src = EMBED_MC, EMBED_CAL, 'embedded reference data'
elif DATA_MODE in ('production', 'quick'):
    suf = '' if DATA_MODE == 'production' else '_quick'
    paths = [f'{RESULTS}/d4_noisy_mc{suf}.csv', f'{RESULTS}/surface_code_plog{suf}.csv']
    missing = [x for x in paths if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError(f'DATA_MODE={DATA_MODE!r}: {missing} not found. Run D4_noisy_simulation.ipynb with '
                                f'QUICK={DATA_MODE == "quick"} (and upload its results/ here), or set DATA_MODE = "reference".')
    mc_text, cal_text, src = open(paths[0]).read(), open(paths[1]).read(), ' + '.join(paths)
else:
    raise ValueError(DATA_MODE)
mc, cal = pd.read_csv(io.StringIO(mc_text)), pd.read_csv(io.StringIO(cal_text))
OUT_SUFFIX = '_quick' if DATA_MODE == 'quick' else ''
PROV = dict(data_mode=DATA_MODE, mc_sha256=sha(mc_text), cal_sha256=sha(cal_text),
            mc_run_ids=';'.join(sorted(map(str, mc.run_id.unique()))) if 'run_id' in mc else 'unknown',
            cal_run_ids=';'.join(sorted(map(str, cal.run_id.unique()))) if 'run_id' in cal else 'unknown',
            mc_shots=int(mc.shots.sum()))
print('source:', src)
for k, v in PROV.items():
    print(f'  {k} = {v}')
print('  identical to the embedded reference:', PROV['mc_sha256'] == REF_SHA['mc'] and PROV['cal_sha256'] == REF_SHA['cal'])
print('  python', platform.python_version(), '| numpy', np.__version__, '| stim', stim.__version__, '| pymatching', pymatching.__version__)
# the same noise convention on both sides: p_idle / p per scenario must agree
for s, f in (('A', 1.0), ('B', 0.0)):
    for name, df in (('MC', mc), ('calibration', cal)):
        r_ = df[df.scenario == s]
        assert np.allclose(r_.p_idle, f * r_.p), f'{name}: scenario {s} has p_idle != {f} p'
print(len(mc), 'Monte Carlo chunks,', len(cal), 'calibration points')
'''),
    md(r'''
## 第I部 表面符号の論理誤り率の較正（Litinski 方式の入力）
1 ラウンドあたりの論理誤り（X と Z の和）$p_L$ を $p_L = A\,(p/p_{\rm th})^{(d+1)/2}$ に当てはめる（GoSC 式 (10) と同じ形。GoSC は $A=0.1$, $p_{\rm th}=0.01$）。閾値より十分下の $p\le0.004$ で、失敗が 20 回以上ある点を使う。$p_{\rm th}$ は仮定した関数形の当てはめの係数で、有限サイズスケーリングで求めた閾値ではない。測定は $p=0.002$〜$0.006$ なので、$p\le10^{-3}$ の $p_L$ は**この関数形による外挿**である。不確かさは各点の失敗数の二項分布による再標本化（`N_BOOT` 回）の 16〜84% 範囲。
'''),
    code(r'''
PLOG = {}
rows = []
rng_b = np.random.default_rng(SEED)
for s in SCENARIOS:
    c_ = cal[(cal.scenario == s) & (cal.pL_per_round > 0)]
    w_ok = c_[((c_.PL_x + c_.PL_z) * c_.shots >= 20) & (c_.p <= 0.004)]
    A, pth = fit_plog(list(zip(w_ok.d, w_ok.p, w_ok.pL_per_round)))
    PLOG[s] = make_plog(A, pth) if PLOG_SOURCE == 'calibrated' else plog_gosc
    bs = []
    for b in range(N_BOOT):
        px = rng_b.binomial(w_ok.shots, w_ok.PL_x) / w_ok.shots
        pz = rng_b.binomial(w_ok.shots, w_ok.PL_z) / w_ok.shots
        r_ = w_ok.rounds
        pl = (1 - (1 - 2 * px) ** (1 / r_)) / 2 + (1 - (1 - 2 * pz) ** (1 / r_)) / 2
        ok = pl > 0
        bs.append(fit_plog(list(zip(w_ok.d[ok], w_ok.p[ok], pl[ok]))))
    bs = np.array(bs)
    pl11 = [float(make_plog(a_, t_)(1e-3, 11)) for a_, t_ in bs]
    rows.append(dict(scenario=s, A=A, A_16=np.quantile(bs[:, 0], 0.16), A_84=np.quantile(bs[:, 0], 0.84),
                     p_th=pth, p_th_16=np.quantile(bs[:, 1], 0.16), p_th_84=np.quantile(bs[:, 1], 0.84), points=len(w_ok),
                     pL_d11_p1e3=float(make_plog(A, pth)(1e-3, 11)), pL_d11_p1e3_16=np.quantile(pl11, 0.16),
                     pL_d11_p1e3_84=np.quantile(pl11, 0.84)))
pd.DataFrame(rows)
'''),
    md(r'''
## 第II部 Kobayashi 方式
### II-1. モンテカルロの結果
ショットごとの不忠実度（0, 1/2, 1）の平均 $\hat\varepsilon_K$ と標準誤差（SE、95% 信頼区間ではない）。集計は全ての条件（シナリオ、$d$、$p$、$p_{\rm idle}$、$\kappa_0,\kappa_{\rm in},\kappa_{\rm out}$、`window`）ごとで、異なる条件は混ぜない。
* `window = noisy`：本来のシミュレーション（全ての位置に雑音）。
* `window = quiet`：**診断用の別の実験**。切り替えの時間窓（D4 相最後のラウンド、$U$ 直前の $Z$ ラウンド、$U$、switch-out、FSC に戻った最初のラウンド）だけ雑音を 0 にし、**同じ復号器**で評価したもの。本研究の復号器が 1 個の故障を訂正できない場所の大部分がこの窓にある（`validation.ipynb` N5）ので、「窓の外の誤りがどれだけ抑えられるか」を見る。**最適な復号器の性能の限界（上界・下界）ではない**：最適な復号器は窓の外の誤りも改善しうるし、窓の誤りの影響が 0 になるとも限らない。
* $\kappa_{\rm in}\neq1$ の行は switch-in のラウンド数の感度。
'''),
    code(r'''
KEYS = ['window', 'scenario', 'd', 'p', 'p_idle', 'kappa0', 'kappa_in', 'kappa_out']
g = mc.groupby(KEYS).agg(shots=('shots', 'sum'), s=('sum_infidelity', 'sum'), n_fail=('n_fail', 'sum'),
                         n_half=('n_half', 'sum'), errors_per_shot=('mean_errors', 'mean')).reset_index()
g['eps'] = g.s / g.shots
n_one = g.n_fail - g.n_half
g['stderr'] = np.sqrt(np.maximum((n_one + 0.25 * g.n_half) / g.shots - g.eps ** 2, 0) / g.shots)
g['eps_over_p'] = g.eps / g.p
k1 = (g.kappa0 == 1) & (g.kappa_in == 1) & (g.kappa_out == 1)
main = g[k1 & (g.window == 'noisy')].copy()
quiet = g[k1 & (g.window == 'quiet')].copy()
g.round({'eps': 6, 'stderr': 6, 'errors_per_shot': 1, 'eps_over_p': 2})
'''),
    md(r'''
### II-2. $d$ と $p$ への依存のモデル（外挿の根拠）
データの構造（II-1）：noisy では $d\ge5$ で $\varepsilon_K$ が $d$ でほとんど下がらない。quiet（窓の雑音なし）では $d$ とともに下がる。そこで次の**モデル（仮定）**を使う。
* **bulk**（窓の外の寄与）：quiet のデータに $\varepsilon_{\rm bulk}=\exp[a + c\,d + (b + e\,d)\log p]$。
* **窓の項**：noisy のデータ（$d\ge5$）に $\varepsilon_K=\varepsilon_{\rm bulk}+W$、$W(p)=w\,p^{\gamma}$（$d$ によらないと**仮定**）。感度を見るため $W=w\,p^{\gamma}e^{\delta(d-7)}$（$d$ に依存しうる）も当てはめる。
* 当てはめは最尤法：各点の不忠実度の和 $s$（1/2 単位で数えた $2s$）を、平均 $2\,\mathrm{shots}\,\varepsilon$ の Poisson 分布として扱う。**失敗 0 の点も含める**（失敗の多い点だけの対数回帰にしない）。
* **当てはめの可否の判定**：bulk の点について、設計行列の階数 4、異なる $d$ が 3 個以上、異なる $p$ が 2 個以上、失敗のある点が 5 個以上、失敗の合計 20 回以上。満たさなければ**ここで停止し、最小費用は計算しない**（`quick` のデータはこれを満たさない）。
* **不確かさ**：当てはめたモデルから各点の $2s$ を Poisson で再標本化して当てはめ直すパラメトリック・ブートストラップ（`N_BOOT` 回；bulk、窓の項、$d$ 依存の窓の項はそれぞれ自分の当てはめから再標本化）。打ち切り規則（時間または失敗数）の影響は考慮していない。
* **外挿**：$d>9$ は $d$ 方向の外挿、quiet の最小の $p$ 未満は bulk の $p$ 方向の外挿。表では別々の列で示す。
'''),
    code(r'''
from scipy.optimize import minimize
def _design_bulk(df):
    return np.stack([np.ones(len(df)), df.d, np.log(df.p), df.d * np.log(df.p)], axis=1)
def fit_bulk(df, y2):
    X_ = _design_bulk(df); ex2 = 2 * df.shots.values
    pos = y2 > 0
    th0, *_ = np.linalg.lstsq(X_[pos], np.log(y2[pos] / ex2[pos]), rcond=None)
    f = lambda th: np.sum(ex2 * np.exp(X_ @ th) - y2 * (X_ @ th))
    return minimize(f, th0, method='BFGS').x
def bulk_eps(th, d, p):
    return np.exp(th[0] + th[1] * d + (th[2] + th[3] * d) * np.log(p))
def win_eps(t, d, p):
    return np.exp(t[0] + t[1] * np.log(p) + (t[2] * (d - 7) if len(t) == 3 else 0))
def fit_window(df, y2, th_b, alt=False):
    ex2 = 2 * df.shots.values; b_ = bulk_eps(th_b, df.d.values, df.p.values)
    def f(t):
        mu = ex2 * (b_ + win_eps(t, df.d.values, df.p.values))
        return np.sum(mu - y2 * np.log(mu))
    t0 = np.array([np.log(5.0), 1.0] + ([0.0] if alt else []))
    return minimize(f, t0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9)).x

def identifiable(df):
    X_ = _design_bulk(df)
    return dict(points=len(df), rank=int(np.linalg.matrix_rank(X_)) if len(df) else 0, distinct_d=df.d.nunique(),
                distinct_p=df.p.nunique(), points_with_failures=int((df.n_fail > 0).sum()), failures=int(df.n_fail.sum()))
CHECK = {s: identifiable(quiet[quiet.scenario == s]) for s in SCENARIOS}
print(pd.DataFrame(CHECK).T)
bad = [s for s, c_ in CHECK.items() if not (c_['rank'] == 4 and c_['distinct_d'] >= 3 and c_['distinct_p'] >= 2
                                           and c_['points_with_failures'] >= 5 and c_['failures'] >= 20)]
if bad:
    raise RuntimeError(f'not enough data to identify the bulk model for scenario(s) {bad} (DATA_MODE={DATA_MODE!r}): '
                       'the analysis stops here (aggregated results above); minimum costs are not computed.')

FITS = {}
rows = []
rng_b = np.random.default_rng(SEED + 1)
q16 = lambda v: np.quantile(v, 0.16)
q84 = lambda v: np.quantile(v, 0.84)
supp = lambda th, p: np.exp(-2 * (th[1] + th[3] * np.log(p)))
for s in SCENARIOS:
    q_ = quiet[quiet.scenario == s]; n_ = main[(main.scenario == s) & (main.d >= 5)]
    yq, yn = 2 * q_.s.values, 2 * n_.s.values
    th_b = fit_bulk(q_, yq); t_w = fit_window(n_, yn, th_b); t_a = fit_window(n_, yn, th_b, alt=True)
    boot = []
    muq = 2 * q_.shots.values * bulk_eps(th_b, q_.d.values, q_.p.values)
    mun = 2 * n_.shots.values * (bulk_eps(th_b, n_.d.values, n_.p.values) + win_eps(t_w, n_.d.values, n_.p.values))
    mua = 2 * n_.shots.values * (bulk_eps(th_b, n_.d.values, n_.p.values) + win_eps(t_a, n_.d.values, n_.p.values))
    for b in range(N_BOOT):                  # each model is resampled from its own fit
        yq_b = rng_b.poisson(muq).astype(float)
        yn_b, ya_b = rng_b.poisson(mun).astype(float), rng_b.poisson(mua).astype(float)
        tb = fit_bulk(q_, yq_b)
        boot.append((tb, fit_window(n_, yn_b, tb), fit_window(n_, ya_b, tb, alt=True)))
    p_ref = float(q_.p.min())
    FITS[s] = dict(central=(th_b, t_w, t_a), boot=boot, p_min_bulk=p_ref, d_max_data=int(main.d.max()))
    sup_b = [supp(bt[0], p_ref) for bt in boot]
    Wp = [win_eps(bt[1], 7, p_ref) / p_ref for bt in boot]
    dl = [bt[2][2] for bt in boot]
    rows.append(dict(scenario=s, bulk_points=len(q_), window_points=len(n_), p_ref=p_ref,
                     bulk_suppression_per_2d=supp(th_b, p_ref), bulk_supp_16=q16(sup_b), bulk_supp_84=q84(sup_b),
                     W_over_p=win_eps(t_w, 7, p_ref) / p_ref, W_over_p_16=q16(Wp), W_over_p_84=q84(Wp),
                     W_gamma=t_w[1], alt_delta=t_a[2], alt_delta_16=q16(dl), alt_delta_84=q84(dl)))

def eps_K_model(s, d, p, w='noisy', params=None, alt=False):
    th_b, t_w, t_a = params if params is not None else FITS[s]['central']
    v = bulk_eps(th_b, d, p)
    if w == 'noisy':
        v = v + win_eps(t_a if alt else t_w, d, p)
    return float(min(v, 1.0))
pd.DataFrame(rows).round(4)
'''),
    md(r'''
`bulk_suppression_per_2d`：当てはめた bulk のモデルで距離を 2 増やしたときの減少率（$p$ = `p_ref`＝quiet の最小の $p$、測定範囲の端）。`W_over_p`：窓の項 $W/p$（$d=7$、同じ $p$）。`alt_delta`：$W$ の $d$ 依存を許したときの $\delta$（$\delta<0$ なら $d$ とともに減る）。いずれも**当てはめたモデルの値**であり、実測の抑制率ではない。
'''),
    code(r'''
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
mk = {3: 'o', 5: 's', 7: '^', 9: 'D'}
cols = {3: '#8fb8ea', 5: '#5a95dc', 7: '#2a78d6', 9: '#174a8a'}
pp = np.logspace(-4.2, -2.9, 40)
for i, (w_, data) in enumerate((('noisy', main), ('quiet', quiet))):
    for j, s in enumerate(SCENARIOS):
        ax = axes[i, j]
        for d in sorted(data.d.unique()):
            m_ = data[(data.scenario == s) & (data.d == d)].sort_values('p')
            pos = m_[m_.n_fail > 0]; zero = m_[m_.n_fail == 0]
            ax.errorbar(pos.p, pos.eps, yerr=pos.stderr, fmt=mk.get(d, 'o'), color=cols.get(d, 'k'), capsize=2,
                        label=f'd = {d} (data ± SE)')
            if len(zero):
                ax.plot(zero.p, 3 / zero.shots, 'v', color=cols.get(d, 'k'), mfc='none')
            ax.plot(pp, [eps_K_model(s, d, x, w_) for x in pp], '-', color=cols.get(d, 'k'), lw=1, alpha=0.7)
        ax.plot(pp, pp, ':', color='#52514e', lw=1, label='error = p')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'[{DATA_MODE}] scenario {s}, ' + ('full simulation' if w_ == 'noisy' else
                     'diagnostic: switching window noise-free'), fontsize=9)
        if i == 1: ax.set_xlabel('physical error rate p')
        if j == 0: ax.set_ylabel('output infidelity  ε_K(d, p)')
axes[0, 1].legend(fontsize=7, title='lines: fitted model (extrapolated outside the data)\nopen triangles: no failure, 3/shots',
                  title_fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIGURES}/d4_logical_error{OUT_SUFFIX}.png', dpi=150); plt.show()
'''),
    md(r'''
### II-3. 資源（同じ回路から数える）
量子ビット数（ピーク）と時間ステップ数を、シミュレーションに使ったのと同じ回路（$\kappa_0=\kappa_{\rm in}=\kappa_{\rm out}=1$）から $d=3,\dots,11$ で数え、$d$ の多項式に当てはめて大きな $d$ に延長する（$d=3..11$ で直接数えた値と一致することを確認）。体積は「各ブロックで使う量子ビット数 × そのブロックの時間ステップ数」の和（雑音なしの最終ラウンドは含めない）。switch-out は $g$ の読み出し 1 ステップ＋補正 CZ の固定スケジュールの層数（測定結果によらない最悪値。シミュレーションの仮想的な遅延測定のモーメントは時間に数えない：`block_steps`）。
**含めていない費用**：FSC から通常の回転表面符号パッチへの変換（未構成、[KZH] にも記述なし）、古典処理の待ち時間、フィードフォワードの $X$ 補正（雑音なし・時間 0 の理想化）。一方、測定スケジュールは最適化していない（FSC 14 ステップ/ラウンド、表面符号は 6）。前者は費用の過小評価、後者は過大評価の要因で、全体として一方向に「保守的」とは言えない。
'''),
    code(r'''
def kobayashi_counts(d):
    circ = D4SwitchingCircuit(d, d, d, d)
    vol, steps = 0, 0
    for blk in circ.blocks:
        if blk['name'][0] == 'FIN':
            continue
        code_ = blk['code']
        if code_ == 'FSC':
            active = set(circ.data_rb) | set(circ.anc['FSC'])
        elif code_ == 'D4':
            active = set(circ.data_all) | set(circ.anc['D4'])
        else:
            active = set(circ.data_all)                   # U and the switch-out
        n_steps = block_steps(circ, blk)
        vol += len(active) * n_steps
        steps += n_steps
    return dict(d=d, qubits=circ.nq, time_steps=steps, volume=vol, switch_out_steps=circ.so_steps)
KC = pd.DataFrame([kobayashi_counts(d) for d in (3, 5, 7, 9, 11)])
POLY = {k: np.polyfit(KC.d, KC[k], 3 if k == 'volume' else 2) for k in ('qubits', 'time_steps', 'volume')}
def kobayashi_cost(d):
    return {k: float(np.polyval(POLY[k], d)) for k in POLY}
KC['volume_poly'] = [kobayashi_cost(d)['volume'] for d in KC.d]
assert np.allclose(KC.volume_poly, KC.volume) and np.allclose([kobayashi_cost(d)['time_steps'] for d in KC.d], KC.time_steps)
print({k: np.round(v, 2) for k, v in POLY.items()})
KC
'''),
    md(r'''
## 第III部 Litinski 方式（誤り予算モデル）
* **GoSC 15-to-1**（Fig. 17c：11 tiles × 11τ、1τ = $d$ cycles）：単一の距離 $d$。
* **GoSC 225-to-1**（Fig. 19：176 tiles × 15τ）：1 段目 $d_1$、2 段目 $d_2$。
* **Litinski 2019 15-to-1** $(d_X,d_Z,d_m)$ と **(15-to-1)×(15-to-1)**：Litinski（2019）の配置と費用式（公開コードと同じ）。
* 入力の魔法状態：確率 $p$ の Pauli 誤り（GoSC Sec. 4 の仮定）。出力誤りは事後選択後の不忠実度、時間は成功確率で割る（期待値）。
* 量子ビット数はデータ＋測定量子ビット（1 tile = $2d^2$）、時間は code cycle × `STEPS_PER_CYCLE`。
* 以下の「最小」は、この格子探索の**候補集合の中の最小**である。
'''),
    code(LIT_CODE),
    md(r'''
## 第IV部 比較
### IV-1. 出力誤り vs 時空間体積
Litinski 方式は候補の中のパレート前線。Kobayashi 方式は $d=3,\dots,9$ がシミュレーション（マーカー）、線は II-2 のモデル（$d>9$ は外挿）：実線＝本来のシミュレーションのモデル（bulk＋窓の項）、点線＝診断（窓の雑音なし、bulk のみ）。点線は**限界ではなく**、「窓の誤りが無かったら」という診断である。
'''),
    code(r'''
def pareto(df):
    df = df.sort_values('volume_steps')
    best, rows_ = np.inf, []
    for _, r in df.iterrows():
        if r.eps < best:
            best = r.eps; rows_.append(r)
    return pd.DataFrame(rows_)
show_p = [1e-4, 3e-4, 1e-3]
fig, axes = plt.subplots(2, len(show_p), figsize=(4.2 * len(show_p), 7.6), sharex=True, sharey=True)
ds = np.arange(3, 61, 2)
vk = np.array([kobayashi_cost(d)['volume'] for d in ds])
for i, s in enumerate(SCENARIOS):
    for j, p in enumerate(show_p):
        ax = axes[i, j]
        fr = pareto(LIT[(s, p)])
        ax.step(fr.volume_steps, fr.eps, where='post', color=C_LIT, label='Litinski (best in the candidate set)')
        ax.plot(vk, [eps_K_model(s, d, p) for d in ds], '-', color=C_KOB, lw=1.2, label='Kobayashi, model (d > 9 extrapolated)')
        ax.plot(vk, [eps_K_model(s, d, p, 'quiet') for d in ds], ':', color=C_KOB, lw=1.5,
                label='Kobayashi, diagnostic: window noise-free')
        sim = main[(main.scenario == s) & (main.p == p)]
        ax.errorbar([kobayashi_cost(d)['volume'] for d in sim.d], sim.eps, yerr=sim.stderr, fmt='o', color=C_KOB,
                    label='Kobayashi, simulated (± SE)')
        ax.axhline(p, color='#52514e', ls=':', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-14, 1)
        ax.set_title(f'[{DATA_MODE}] scenario {s},  p = {p:g}', fontsize=10)
        if j == 0: ax.set_ylabel('output error of one T state')
        if i == 1: ax.set_xlabel('space-time volume [physical qubits x time steps]')
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIGURES}/cost_comparison{OUT_SUFFIX}.png', dpi=150); plt.show()
'''),
    md(r'''
### IV-2. 要求誤りごとの最小費用
* Litinski 方式：要求を満たす候補のうち体積最小のもの（候補集合内の最小）。
* Kobayashi 方式：要求を満たす最小の $d$（$d\le61$ の範囲で探索）。`kobayashi_*` は本来のシミュレーションのモデル（bulk＋窓の項）、`kobayashi_diag_*` は診断（窓の雑音なし）、`kobayashi_altW_*` は窓の項の $d$ 依存を許したモデル（感度）。
* `status`：ブートストラップの複製のうち、$d\le61$ で要求に届く割合が 84% 以上なら `reached`、16% 以下なら `not_reached_model`（このモデル・探索範囲では届かない。方式として不可能という意味ではない）、その間は `unresolved`。体積の `_16` / `_84` は届いた複製の体積の 16〜84% 範囲。
* `d_extrapolated`：選ばれた $d$ がシミュレーションした最大の $d$ を超える（$d$ 方向の外挿）。`p_extrapolated`：$p$ が bulk の当てはめに使った最小の $p$ より小さい（$p$ 方向の外挿）。外挿でなくても値はモデルによる推定であり、その要求誤りを直接観測したものではない。
* 各行に入力データの来歴（`data_mode`, `mc_sha256`, `cal_sha256`, `mc_run_ids`, ...）を記録する。
'''),
    code(r'''
def dmin(s, p, t, w_='noisy', params=None, alt=False):
    return next((d for d in range(3, 62, 2) if eps_K_model(s, d, p, w_, params, alt) <= t), None)
def kob_entry(s, p, t, w_, alt=False):
    dc = dmin(s, p, t, w_, alt=alt)
    db = [dmin(s, p, t, w_, bt, alt) for bt in FITS[s]['boot']]
    reach = [d for d in db if d is not None]
    frac = len(reach) / len(db)
    status = 'reached' if frac >= 0.84 else 'not_reached_model' if frac <= 0.16 else 'unresolved'
    vols = [kobayashi_cost(d)['volume'] for d in reach]
    kc = kobayashi_cost(dc) if dc else None
    return dict(status=status, frac_reached=frac, d=dc if dc else '—',
                d_extrapolated=bool(dc is not None and dc > FITS[s]['d_max_data']),
                qubits=kc['qubits'] if kc else np.nan, time_steps=kc['time_steps'] if kc else np.nan,
                volume=kc['volume'] if kc else np.nan,
                volume_16=np.quantile(vols, 0.16) if vols else np.nan, volume_84=np.quantile(vols, 0.84) if vols else np.nan)
rows = []
for s in SCENARIOS:
    for p in P_EVAL:
        df = LIT[(s, p)]
        for t in TARGETS:
            ok = df[df.eps <= t]
            lit = ok.loc[ok.volume_steps.idxmin()] if len(ok) else None
            r = dict(scenario=s, p=p, target=t, p_extrapolated=bool(p < FITS[s]['p_min_bulk']),
                     litinski_protocol=lit.protocol if lit is not None else 'none in candidate set',
                     litinski_eps=lit.eps if lit is not None else np.nan,
                     litinski_qubits=lit.qubits if lit is not None else np.nan,
                     litinski_time_steps=lit.time_steps if lit is not None else np.nan,
                     litinski_volume=lit.volume_steps if lit is not None else np.nan)
            for pre, w_, alt in (('kobayashi', 'noisy', False), ('kobayashi_diag', 'quiet', False),
                                 ('kobayashi_altW', 'noisy', True)):
                r.update({f'{pre}_{k}': v for k, v in kob_entry(s, p, t, w_, alt).items()})
            rows.append(r)
CMP = pd.DataFrame(rows)
for pre in ('kobayashi', 'kobayashi_diag', 'kobayashi_altW'):
    CMP[f'ratio_{pre}_over_litinski'] = CMP[f'{pre}_volume'] / CMP.litinski_volume
for k, v in PROV.items():
    CMP[k] = v
pd.set_option('display.max_colwidth', 40)
CMP[['scenario', 'p', 'target', 'litinski_protocol', 'litinski_volume', 'kobayashi_status', 'kobayashi_volume',
     'kobayashi_altW_status', 'kobayashi_diag_status', 'kobayashi_diag_d', 'kobayashi_diag_d_extrapolated', 'p_extrapolated',
     'ratio_kobayashi_diag_over_litinski', 'kobayashi_diag_volume_16', 'kobayashi_diag_volume_84']]
'''),
    md(r'''
### IV-3. Kobayashi 方式が同じ体積に収まるために必要な改善（診断値）
要求誤りごとに、Litinski 方式の最小体積以下に収まる Kobayashi 方式の最大の $d$（$d_{\max}$）を求め、そこでの $\varepsilon_K(d_{\max},p)$（本来のシミュレーションのモデル）が要求誤りの何倍か（`required_improvement`）を示す。**現在の費用モデルを固定した場合の目安**であり、復号器やスケジュールを変えれば量子ビット数・時間も変わるので、この倍率の改善で必ず同じ費用に並ぶという保証ではない。
'''),
    code(r'''
rows = []
for _, r in CMP.iterrows():
    if not np.isfinite(r.litinski_volume):
        continue
    ds_ = [d for d in range(3, 62, 2) if kobayashi_cost(d)['volume'] <= r.litinski_volume]
    dmax = max(ds_) if ds_ else None
    ek = eps_K_model(r.scenario, dmax, r.p) if dmax else np.nan
    rows.append(dict(scenario=r.scenario, p=r.p, target=r.target, litinski_volume=r.litinski_volume,
                     kobayashi_d_max=dmax, eps_K_at_d_max=ek, required_improvement=ek / r.target if dmax else np.nan))
IMP = pd.DataFrame(rows)
CMP = CMP.merge(IMP[['scenario', 'p', 'target', 'kobayashi_d_max', 'required_improvement']], how='left',
                on=['scenario', 'p', 'target'])
CMP.to_csv(f'{RESULTS}/cost_comparison{OUT_SUFFIX}.csv', index=False)
print('wrote', f'{RESULTS}/cost_comparison{OUT_SUFFIX}.csv', '| data_mode =', DATA_MODE)
IMP.round({'eps_K_at_d_max': 8, 'required_improvement': 1})
'''),
    md(CONCLUSION),
]
save('noisy_cost_comparison.ipynb', cells)
