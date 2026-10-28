# 物理誤り率を与えたときの、D4 code switching（Kobayashi 方式）と 15-to-1 蒸留（Litinski 方式）の資源比較（v5）

論理 $T$ 魔法状態 1 個を、要求される出力誤り以下で作るのに必要な時空間体積（物理量子ビット × 時間ステップ）を、**同じ回路レベル雑音モデル**（物理誤り率 $p$）の下で比べる。

* **Kobayashi 方式**：folded surface code → D4 Clifford 安定化子符号への code switching → transversal $U$（論理 $T^\dagger$）→ 復帰（Kobayashi–Zhu–Hsin, arXiv:2511.02900）。非 Clifford 回路（CCZ・CS・$T^\dagger$ を含む）の雑音付きシミュレーションは **Takada–Bartlett–Williamson（arXiv:2609.16929）の Pauli proxy 法を採用**し（適用条件は D4 code switching について検証済み）、出力誤り $\varepsilon_K(d,p)$ をモンテカルロで求めた。
* **Litinski 方式**：15-to-1 蒸留（A Game of Surface Codes の 11-tile／176-tile 配置と、Litinski 2019 の最適化された蒸留器）。Litinski 2019 の誤り予算モデル（公開コードを再実装）に、同じ雑音モデルで較正した表面符号の論理誤り率を入れた。

目的・設定・手法（Takada の方法をどう使ったか、非 Clifford シミュレーションの仕組み、復号器）・仮定・検証の範囲は **[`docs/methods.md`](docs/methods.md)** にまとめた。

## ファイル構成と実行順

| 順 | ノートブック | 役割 | 所要時間 |
|---|---|---|---|
| 1 | `D4_noisy_simulation.ipynb` | Kobayashi 方式の雑音付きモンテカルロ（$\varepsilon_K(d,p)$）と表面符号の $p_L$ の較正 → `results/d4_noisy_mc.csv`, `results/surface_code_plog.csv` | `QUICK=True`（既定）で数分の動作確認（`*_quick.csv` に出力し本番結果を上書きしない）、本番は 4 コアで約 5 時間（診断用の点を含む） |
| 2 | `noisy_cost_comparison.ipynb` | **比較の本体**：$\varepsilon_K$ の当てはめ、Litinski 方式の誤り予算、要求誤りごとの最小費用、図 → `results/cost_comparison.csv`, `figures/*.png` | 約 5 分 |
| 3 | `validation.ipynb` | 正しさの検証（v4 の D4・15-to-1 の検証＋v5 の雑音付きシミュレーションの検証）→ `results/validation.json` | 約 30 分 |

* 各ノートブックは単独で実行できる（共通コードを各ノートに埋め込んである）。2 は 1 の結果ファイルが無ければ、ノートに埋め込まれた本番結果を使う。
* 出力は `results/`（CSV 3 個＋JSON 1 個）と `figures/`（PNG 2 個）だけ。乱数シードは `SEED = 20261024`。

### Google Colab での実行
ノートブックを開いて上から順に実行する。`stim`・`pymatching` などが無ければ 2 番目のセルが自動でインストールする。出力はランタイム終了で消えるので、残す場合は最初のセルのコメントに従って Google Drive をマウントする。

### 共通コードの保守（研究の実行には不要）
埋め込まれた共通コードの元は `tools/` にある（`common_lib.py`, `d4_lib.py`, `msd_lib.py`, `noisy_d4_lib.py`, `msd_noisy_lib.py`）。変更する場合は `tools/` のソースを編集して `python tools/build_all.py` を実行すると、3 つのノートブックが同じ埋め込みコードで再生成される（実行結果は消えるので、生成後に実行する）。`build_sim.py`, `build_main.py`, `build_val.py` が各ノートブックの生成スクリプト。

## 主な結果

数値は `results/`、図は `figures/`、詳しい解釈は `noisy_cost_comparison.ipynb` の結論と `docs/methods.md` §8。**Kobayashi 方式の数値は本研究で構成した復号器・測定スケジュールに依存する**（[KZH] が引用する JIT decoder は参照できなかった）。

**1. Kobayashi 方式の出力誤り $\varepsilon_K$（モンテカルロ、$d=3..9$、約 176 万ショット＋診断 59 万ショット）**
* 本来のシミュレーションでは、$d\ge5$ で $\varepsilon_K$ が $d$ によらずほぼ $(3\text{–}8)\,p$（アイドル雑音なし B）にとどまり、生の注入状態（誤り $\sim p$）より悪い。アイドル雑音あり（A）では $p=10^{-3}$ で約 20%。
* 原因：切り替えの時間窓（D4 相最後のラウンド〜$U$〜switch-out〜FSC の最初のラウンド）で、本研究の復号器が 1 個の故障を訂正できない場所が一定数ある（$d=3$ で全故障の 0.39%、`validation.ipynb` N5）。
* この窓だけ雑音を 0 にした**診断**では $\varepsilon_K$ は $d$ とともに急速に下がる（B、$p=10^{-4}$ で距離 2 ごとに約 24 倍；A は約 15 倍、$p=10^{-3}$ では閾値付近）。実際の性能は、最適な復号器なら両者の間にあると考えられる。

**2. 要求誤りごとの最小体積の比（Kobayashi / Litinski）**（Litinski 方式は GoSC と Litinski 2019 の蒸留器のうち最小のもの、$p_L$ は同じ雑音で較正）

| | 要求 $10^{-4}$ | 要求 $10^{-6}$ | 要求 $10^{-8}$ | 要求 $10^{-10}$ |
|---|---|---|---|---|
| 本来のシミュレーション（A, B、全ての $p$） | 届かない | 届かない | 届かない | 届かない |
| 楽観的な限界、A、$p=10^{-4}$ / $3\times10^{-4}$ / $10^{-3}$ | 1.6 / 5.5 / 届かない | 5.6 / 25 / 届かない | 7.1 / 17 / 届かない | 6.0 / 5.1 / 届かない |
| 楽観的な限界、B、$p=10^{-4}$ / $3\times10^{-4}$ / $10^{-3}$ | 0.52 / 1.6 / 24 | 1.0 / 5.6 / 58 | 5.6 / 6.0 / 24 | 5.4 / 1.8 / 届かない |

（Litinski 方式の最小体積は、例えば $p=10^{-4}$・要求 $10^{-8}$ で約 $1.2\times10^5$ 量子ビット・ステップ（(15-to-1)$_{9,3,3}$）、$p=10^{-3}$・要求 $10^{-10}$ で約 $1.5\times10^7$（2 段蒸留）。）

**3. 結論**：本研究の範囲では、実用的な要求精度（$10^{-6}$ 以下）で魔法状態 1 個を作る時空間体積は **Litinski 方式（15-to-1 蒸留）の方が小さい**。切り替えの窓を理想的に扱えたとしても多くの条件で 2〜60 倍の差が残る。Kobayashi 方式が並ぶのは、アイドル雑音が無く、$p\le2\times10^{-4}$ で要求が緩い（$10^{-4}$〜$10^{-6}$）場合だけ。結論を変えうる要因（窓を扱う JIT 復号器、スケジュールの最適化、CCZ の実際の誤り率、FSC→パッチ変換）は `docs/methods.md` §7–8。

## 版
v4（雑音 0 での回路構造と資源の比較）は git の履歴（コミット `049ce8f`）に残してある。v5 での変更は `docs/methods.md` §9。
