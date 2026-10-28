# 物理誤り率を与えたときの、D4 code switching（Kobayashi 方式）と 15-to-1 蒸留（Litinski 方式）の資源比較（v5）

論理 $T$ 魔法状態 1 個を、要求される出力誤り以下で作るのに必要な時空間体積（物理量子ビット × 時間ステップ）を、**同じ回路レベル雑音モデル**（物理誤り率 $p$）の下で比べる。

* **Kobayashi 方式**：folded surface code → D4 Clifford 安定化子符号への code switching → transversal $U$（論理 $T^\dagger$）→ 復帰（Kobayashi–Zhu–Hsin, arXiv:2511.02900）。非 Clifford 回路（CCZ・CS・$T^\dagger$ を含む）の雑音付きシミュレーションは **Takada–Bartlett–Williamson（arXiv:2609.16929）の Pauli proxy 法を採用**し（適用条件は D4 code switching について検証済み）、出力誤り $\varepsilon_K(d,p)$ をモンテカルロで求めた。
* **Litinski 方式**：15-to-1 蒸留（A Game of Surface Codes の 11-tile／176-tile 配置と、Litinski 2019 の最適化された蒸留器）。Litinski 2019 の誤り予算モデル（公開コードを再実装）に、同じ雑音モデルで較正した表面符号の論理誤り率を入れた。

目的・設定・手法（Takada の方法をどう使ったか、非 Clifford シミュレーションの仕組み、復号器）・仮定・検証の範囲は **[`docs/methods.md`](docs/methods.md)** にまとめた。

## ファイル構成と実行順

| 順 | ノートブック | 役割 | 所要時間 |
|---|---|---|---|
| 1 | `D4_noisy_simulation.ipynb` | Kobayashi 方式の雑音付きモンテカルロ（$\varepsilon_K(d,p)$）と表面符号の $p_L$ の較正 → `results/d4_noisy_mc.csv`, `results/surface_code_plog.csv` | `QUICK=True`（既定）で数分の動作確認（`*_quick.csv` に出力し本番結果を上書きしない）、本番は 4 コアで約 4 時間 |
| 2 | `noisy_cost_comparison.ipynb` | **比較の本体**：$\varepsilon_K$ の当てはめ、Litinski 方式の誤り予算、要求誤りごとの最小費用、図 → `results/cost_comparison.csv`, `figures/*.png` | 約 5 分 |
| 3 | `validation.ipynb` | 正しさの検証（v4 の D4・15-to-1 の検証＋v5 の雑音付きシミュレーションの検証）→ `results/validation.json` | 約 30 分 |

* 各ノートブックは単独で実行できる（共通コードを各ノートに埋め込んである）。2 は 1 の結果ファイルが無ければ、ノートに埋め込まれた本番結果を使う。
* 出力は `results/`（CSV 3 個＋JSON 1 個）と `figures/`（PNG 2 個）だけ。乱数シードは `SEED = 20261024`。

### Google Colab での実行
ノートブックを開いて上から順に実行する。`stim`・`pymatching` などが無ければ 2 番目のセルが自動でインストールする。出力はランタイム終了で消えるので、残す場合は最初のセルのコメントに従って Google Drive をマウントする。

### 共通コードの保守（研究の実行には不要）
埋め込まれた共通コードの元は `tools/` にある（`common_lib.py`, `d4_lib.py`, `msd_lib.py`, `noisy_d4_lib.py`, `msd_noisy_lib.py`）。変更する場合は `tools/` のソースを編集して `python tools/build_all.py` を実行すると、3 つのノートブックが同じ埋め込みコードで再生成される（実行結果は消えるので、生成後に実行する）。`build_sim.py`, `build_main.py`, `build_val.py` が各ノートブックの生成スクリプト。

## 主な結果

@@RESULTS@@

## 版
v4（雑音 0 での回路構造と資源の比較）は git の履歴（コミット `049ce8f`）に残してある。v5 での変更は `docs/methods.md` §9。
