"""Builds D4_noisy_simulation.ipynb: Monte Carlo of the noisy D4 code switching protocol + surface-code calibration."""
from nbutil import *

cells = [md(r'''
# D4_noisy_simulation — 雑音付きモンテカルロ（Kobayashi 方式の出力誤り・表面符号の論理誤り率）

**目的**：物理誤り率 $p$ の回路レベル雑音の下で、
1. Kobayashi–Zhu–Hsin（[KZH]）の D4 code switching が出力する論理魔法状態 $T^\dagger|+\rangle_L$ の誤り（不忠実度）$\varepsilon_K(d,p)$ を求める（非 Clifford 回路の厳密シミュレーション）。
2. 比較相手（Litinski の蒸留）の誤り予算に使う回転表面符号の論理誤り率 $p_L(d,p)$ を、**同じ雑音モデル**で較正する。

**出力**：`results/d4_noisy_mc.csv`（ショットの塊ごとの生データ）、`results/surface_code_plog.csv`。比較・費用計算は `noisy_cost_comparison.ipynb` が行う。
`QUICK = True`（既定）は数分の動作確認で、結果は `*_quick.csv` に書かれ、リポジトリの本番結果を上書きしない。本番結果（リポジトリ同梱）は `QUICK = False`、4 コアで約 4 時間。

**方法の要点**（詳細は `docs/methods.md`）

| 項目 | 内容 |
|---|---|
| 回路 | v4 で検証済みの D4 符号・測定ガジェット（CCZ を含む）・transversal $U=WV$ を、段階 S0（FSC $|+_L\rangle$ 準備）→ S1（D4 安定化子のラウンド、各ラウンドは Z 検査 → X 型 Clifford 安定化子の順）→ U 直前の Z 検査ラウンド → U → switch-out（$Z^g$ 読み出し）→ S4（FSC ラウンド）→ 雑音なしの最終ラウンド、の順に時間ステップ単位で構成 |
| 雑音 | $k$ 量子ビットゲートの後に $k$ 量子ビット脱分極（確率 $p$、CCZ・CS・$T^\dagger$ も $p$）、初期化後・測定前に反転 $p$、アイドル量子ビットに脱分極 $p_{\rm idle}$。シナリオ A：$p_{\rm idle}=p$、シナリオ B：$p_{\rm idle}=0$（Takada らと同じ） |
| 非 Clifford シミュレーション | **Takada–Bartlett–Williamson の Clifford-stabilizer simulation（Pauli proxy 法, arXiv:2609.16929, Algorithm 2）を採用**。適用条件（Definition 2, Sec. VIII）を D4 code switching について確認し（`validation.ipynb` N1）、さらに $L=1,2$ で非 Clifford 回路の厳密な状態ベクトル計算と全測定で一致することを確認した（N3） |
| 復号 | flux（$Z$ 検査）：毎ラウンドの実時間 MWPM と即時 $X$ 補正（JIT 型、本研究の構成）。charge（$X$ 型検査）：プロトコル終了後に stim の誤りモデル＋MWPM（switch-out をまたぐ検出器の基底を 2 通り試し重みの小さい方を採る）、flux の位置から乱された測定の重みを下げる。[Dav] の JIT decoder は参照できなかったため**本研究独自の復号器**であり、最適ではない |
| 論理誤り | 最後に雑音なしのラウンドで復号した後、出力の論理 Clifford 安定化子 $\bar H$ の期待値から不忠実度 $(1-\langle\bar H\rangle)/2$（Takada Sec. VI D） |
'''),
    code(SETUP), code(INSTALL),
    *embedded_lib_cells('common_lib.py', 'd4_lib.py', 'noisy_d4_lib.py', 'msd_noisy_lib.py'),
    md(r'''
## 1. 設定
* `SCENARIOS`：雑音モデル。A＝アイドル雑音あり（$p_{\rm idle}=p$、SD6 型の標準的な回路雑音）、B＝アイドル雑音なし。
* `KAPPA`：S0, S1, S4 のラウンド数 $\lceil\kappa d\rceil$（[KZH] は $O(d)$ としか書いていない。既定 1、S1 については `KAPPA_SCAN` で感度を見る）。
* `TIME_CAP`（秒）と `MAX_FAIL`：1 点あたりの打ち切り条件（失敗 200 回で十分な精度）。
* `WORKERS`：並列プロセス数（Colab は 2 程度）。
'''),
    code(r'''
QUICK = True
SCENARIOS = {'A': 1.0, 'B': 0.0}            # p_idle = factor * p
P_LIST = [1e-3, 5e-4, 3e-4, 2e-4, 1e-4]
D_LIST = [3, 5, 7, 9]
KAPPA = (1.0, 1.0, 1.0)                     # (kappa0, kappa_in, kappa_out)
KAPPA_SCAN = [(d, 2e-4, 'B', k) for d in (5, 7) for k in (0.4, 0.6, 1.5)]
TIME_CAP, MAX_FAIL, WORKERS = 1200, 200, 4
CALIB_D, CALIB_P = [3, 5, 7, 9, 11], [0.002, 0.003, 0.004, 0.006]
if QUICK:
    P_LIST, D_LIST, KAPPA_SCAN = [1e-3, 3e-4], [3, 5], []
    TIME_CAP, MAX_FAIL, WORKERS = 60, 50, 2
    CALIB_D, CALIB_P = [3, 5, 7], [0.003, 0.006]
SUFFIX = '_quick' if QUICK else ''
MC_CSV, CAL_CSV = f'{RESULTS}/d4_noisy_mc{SUFFIX}.csv', f'{RESULTS}/surface_code_plog{SUFFIX}.csv'
POINTS = [dict(scenario=s, d=d, p=p, p_idle=f * p, kappa=KAPPA) for s, f in SCENARIOS.items() for p in P_LIST for d in D_LIST]
POINTS += [dict(scenario=s, d=d, p=p, p_idle=SCENARIOS[s] * p, kappa=(1.0, k, 1.0)) for d, p, s, k in KAPPA_SCAN]
print(len(POINTS), 'points ->', MC_CSV)
'''),
    md(r'''
## 2. 回路の規模
各 $d$ について、量子ビット数・時間ステップ数・雑音の位置の数（$p$ をかけると 1 ショットあたりの平均誤り数）。D4 相のラウンドは深い（CCZ を含む 3 つの可換クラス＋Z 検査）ため、同じ $d$ の表面符号より誤りの位置がはるかに多い。
'''),
    code(r'''
rows = []
for d in D_LIST:
    R_ = [max(1, int(np.ceil(k * d))) for k in KAPPA]
    circ = D4SwitchingCircuit(d, *R_)
    comp = CompiledProtocol(circ, NoiseModel(1.0, p_idle=1.0))
    blocks = {}
    for b in comp.blocks:
        blocks[b['name'][0]] = blocks.get(b['name'][0], 0) + len(b['moms'])
    rows.append(dict(d=d, qubits=circ.nq, data=circ.nd, time_steps=sum(blocks.values()),
                     **{f'steps_{k}': v for k, v in blocks.items()}, noise_locations=comp.n_loc))
pd.DataFrame(rows)
'''),
    md(r'''
## 3. モンテカルロ（Kobayashi 方式）
1 ショット：誤りを雑音モデルからサンプルし、Pauli proxy 回路（`FastShot`）を stim の `TableauSimulator` で実行する。各ラウンドの後に flux 復号器（`FluxDecoder`）が補正を決め、物理的に適用する（即時フィードフォワード）。最後に charge 復号器（`ChargeDecoder`）の予測と、雑音なしの最終ラウンド後の $\bar H$ の期待値から不忠実度（0, 1/2, 1 のいずれか）を得る。
結果は塊（chunk）ごとに CSV に追記する（途中で止めても集計できる）。
'''),
    code(r'''
import time, multiprocessing as mp_

def run_point(pt, seed):
    d, p = pt['d'], pt['p']
    R_ = [max(1, int(np.ceil(k * d))) for k in pt['kappa']]
    ex = D4Experiment(d, *R_, NoiseModel(p, p_idle=pt['p_idle']))
    rng = np.random.default_rng(seed)
    chunk = 400 if d <= 5 else 100 if d == 7 else 40
    out, t_start, nf = [], time.time(), 0
    while time.time() - t_start < TIME_CAP and nf < MAX_FAIL:
        t0 = time.time()
        inf, st = ex.run(chunk, rng)
        nf += int((inf > 0).sum())
        out.append(dict(scenario=pt['scenario'], d=d, p=p, p_idle=pt['p_idle'], kappa0=pt['kappa'][0],
                        kappa_in=pt['kappa'][1], kappa_out=pt['kappa'][2], R0=R_[0], Rin=R_[1], Rout=R_[2],
                        shots=chunk, sum_infidelity=float(inf.sum()), n_fail=int((inf > 0).sum()),
                        n_half=int((inf == 0.5).sum()), mean_errors=float(np.mean([s['n_err'] for s in st])),
                        seconds=time.time() - t0))
    return out

def _worker(args):
    return run_point(*args)

t0 = time.time()
tasks = [(pt, SEED + 17 * i) for i, pt in enumerate(POINTS)]
if WORKERS > 1:
    with mp_.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(_worker, tasks, chunksize=1)
else:
    res = [_worker(t) for t in tasks]
mc = pd.DataFrame([r for rr in res for r in rr])
mc.to_csv(MC_CSV, index=False)
print('wrote', MC_CSV, len(mc), 'chunks, %.0f s' % (time.time() - t0))
'''),
    md(r'''
## 4. 表面符号の論理誤り率の較正（Litinski 側の $p_L$）
stim が生成する回転表面符号メモリ回路を、D4 と同じ雑音モデル・時間ステップの約束（X 型補助は RX/MX で準備・測定、初期化と測定は別の時間ステップで 1 ラウンド＝初期化＋CNOT 4 層＋測定＝6 時間ステップ、アイドル雑音はシナリオどおり）に置き換え、MWPM で復号する。1 ラウンドあたりの論理誤り（X と Z の和）$p_L$ を求め、`noisy_cost_comparison.ipynb` で $p_L=A\,(p/p_{\rm th})^{(d+1)/2}$ に当てはめる（GoSC 式 (10) と同じ形）。
'''),
    code(r'''
rows = []
t0 = time.time()
for scen, f in SCENARIOS.items():
    for d in CALIB_D:
        for p in CALIB_P:
            shots = (400000 if d <= 5 else 200000 if d <= 7 else 100000) // (20 if QUICK else 1)
            tot, o = 0.0, {}
            for basis in ('x', 'z'):
                c = noisy_surface_code(d, 3 * d, basis, p, p_idle=f * p)
                m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
                det, obs = c.compile_detector_sampler(seed=SEED + d * 1000 + int(p * 1e5)).sample(shots, separate_observables=True)
                PL = float(np.mean(m.decode_batch(det)[:, 0] != obs[:, 0]))
                o[basis] = PL
                tot += (1 - (1 - 2 * PL) ** (1 / (3 * d))) / 2
            rows.append(dict(scenario=scen, d=d, p=p, p_idle=f * p, shots=shots, rounds=3 * d,
                             PL_x=o['x'], PL_z=o['z'], pL_per_round=tot))
cal = pd.DataFrame(rows)
cal.to_csv(CAL_CSV, index=False)
print('wrote', CAL_CSV, '%.0f s' % (time.time() - t0))
cal
'''),
    md(r'''
## 5. 結果の概要
不忠実度の推定値 $\hat\varepsilon_K$ と標準誤差（ショットごとの値 0, 1/2, 1 の平均と標準誤差）。
'''),
    code(r'''
g = mc.groupby(['scenario', 'd', 'p', 'kappa_in']).agg(shots=('shots', 'sum'), s=('sum_infidelity', 'sum'),
                                                       n_fail=('n_fail', 'sum'), n_half=('n_half', 'sum')).reset_index()
g['eps'] = g.s / g.shots
n_one = g.n_fail - g.n_half                      # shots with infidelity 1 (the others have 1/2)
g['stderr'] = np.sqrt(np.maximum((n_one + 0.25 * g.n_half) / g.shots - g.eps ** 2, 0) / g.shots)
g.round(6)
'''),
]
save('D4_noisy_simulation.ipynb', cells)
