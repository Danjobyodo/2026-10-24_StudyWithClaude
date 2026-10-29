# 物理誤り率を与えたときの、D4 code switching（Kobayashi 方式）と 15-to-1 蒸留（Litinski 方式）の資源比較（v5）

論理 $T$ 魔法状態 1 個を、要求される出力誤り以下で作るのに必要な時空間体積（物理量子ビット × 時間ステップ）を、**同じ回路レベル雑音モデル**（物理誤り率 $p$）の下で比べる。

* **Kobayashi 方式**：folded surface code → D4 Clifford 安定化子符号への code switching → transversal $U$（論理 $T^\dagger$）→ 復帰（Kobayashi–Zhu–Hsin, arXiv:2511.02900）。非 Clifford 回路（CCZ・CS・$T^\dagger$ を含む）の雑音付きシミュレーションは **Takada–Bartlett–Williamson（arXiv:2609.16929）の Pauli proxy 法を採用**し（適用条件は D4 code switching について検証済み）、出力誤り $\varepsilon_K(d,p)$ をモンテカルロで求めた。
* **Litinski 方式**：15-to-1 蒸留（A Game of Surface Codes の 11-tile／176-tile 配置と、Litinski 2019 の最適化された蒸留器）。Litinski 2019 の誤り予算モデル（公開コードを再実装）に、同じ雑音モデルで較正した表面符号の論理誤り率を入れた。

目的・設定・手法（Takada の方法をどう使ったか、非 Clifford シミュレーションの仕組み、復号器）・仮定・検証の範囲は **[`docs/methods.md`](docs/methods.md)** にまとめた。

## ファイル構成と実行順

| 順 | ノートブック | 役割 | 所要時間 |
|---|---|---|---|
| 1 | `D4_noisy_simulation.ipynb` | Kobayashi 方式の雑音付きモンテカルロ（$\varepsilon_K(d,p)$）と表面符号の $p_L$ の較正 → `results/d4_noisy_mc.csv`, `results/surface_code_plog.csv`（各行に実行 ID `run_id`） | `QUICK=True`（既定）は数分の動作確認で `*_quick.csv` に出力。本番（`QUICK=False`）は 4 コアで約 5 時間。chunk ごとに保存し、`RESUME=True` で中断から再開できる |
| 2 | `noisy_cost_comparison.ipynb` | **比較の本体**：$\varepsilon_K$ のモデルの当てはめ（不確かさ付き）、Litinski 方式の誤り予算、要求誤りごとの最小費用、図 → `results/cost_comparison.csv`, `figures/*.png` | 約 5 分 |
| 3 | `validation.ipynb` | 正しさの検証 → `results/validation.json` | 約 2.5 時間（`QUICK=True` で短縮） |

* 各ノートブックは単独で実行できる（共通コードを各ノートに埋め込んである）。
* **比較ノートの入力は `DATA_MODE` で明示的に選ぶ**（自動の切り替えはしない）：`'reference'`（既定。ノートに埋め込んだ本番データ＝リポジトリの `results/` と同一で、リポジトリの表・図を再現する）、`'production'`（自分で本番を実行した結果）、`'quick'`（動作確認の結果。点数が足りないのでモデルの当てはめの前で停止する）。入力の SHA-256・実行 ID・ショット数を表示し、比較表の各行にも記録する。
* Colab ではノートごとに別のランタイムなので、`production` / `quick` の結果を比較ノートで使うには、シミュレーションのノートで作られた `results/` の 2 つの CSV を比較ノートのランタイムの `results/` にアップロードする（または両方で同じ Google Drive のフォルダを作業ディレクトリにする）。
* 出力は `results/`（CSV 3 個＋JSON 1 個）と `figures/`（PNG 2 個）だけ。乱数シードは `SEED = 20261024`。

### Google Colab での実行
ノートブックを開いて上から順に実行する。`stim`・`pymatching` などが無ければ 2 番目のセルが自動でインストールする。出力はランタイム終了で消えるので、残す場合は最初のセルのコメントに従って Google Drive をマウントする。

### 共通コードの保守（研究の実行には不要）
埋め込まれた共通コードの元は `tools/` にある（`common_lib.py`, `d4_lib.py`, `msd_lib.py`, `noisy_d4_lib.py`, `msd_noisy_lib.py`）。変更する場合は `tools/` のソースを編集して `python tools/build_all.py` を実行すると、3 つのノートブックが同じ埋め込みコードで再生成される（実行結果は消えるので、生成後に実行する）。`build_sim.py`, `build_main.py`, `build_val.py` が各ノートブックの生成スクリプト。

## 主な結果

本番データ：実行 ID `production-20261029T053925Z-29f79c`（329 万ショット、修正後のコードで 2026-10-29 に実行）。**以前の版（v5.0）の数値は撤回した**（switch-out のモデルを修正したため）。詳しくは `noisy_cost_comparison.ipynb` の結論と `docs/methods.md` §8。**Kobayashi 方式の数値は本研究で構成した回路・スケジュール・復号器での値**であり（[KZH] が引用する JIT decoder は参照できなかった）、Kobayashi 方式一般の性能ではない。

**1. Kobayashi 方式の出力誤り $\varepsilon_K$**
* 本来のシミュレーションでは、$d\ge5$ で $\varepsilon_K$ が $d$ でほとんど下がらず、アイドル雑音なし（B）で $\approx(3\text{–}8)\,p$、アイドル雑音あり（A）ではさらに大きい（$p=10^{-3}$ で約 20%）。
* 原因：切り替えの時間窓（D4 相最後のラウンド〜$U$〜switch-out〜FSC の最初のラウンド）で、この回路・復号器が訂正できない単一故障がある。窓を全列挙した一次の係数（$\varepsilon\approx c_1p$）は $c_1=3.1$（B, $d=5$）、$3.9$（B, $d=7$）で、測定値と整合する（`validation.ipynb` N5）。
* 窓だけ雑音を 0 にした**診断**（同じ復号器）では $d$ とともに下がる（$p=3\times10^{-4}$ で、当てはめたモデルで距離 2 ごとに A 約 4.2 倍、B 約 3.5 倍）。診断は最適な復号器の性能の限界ではない。

**2. 要求誤りごとの最小体積の比（Kobayashi / Litinski）**（Litinski 方式は GoSC と Litinski 2019 の蒸留器の候補集合内の最小、$p_L$ は同じ雑音パラメータで較正）

| | 要求 $10^{-4}$ | 要求 $10^{-6}$ | 要求 $10^{-8}$ | 要求 $10^{-10}$ |
|---|---|---|---|---|
| 本来のシミュレーションのモデル（A, B、全ての $p$） | 届かない | 届かない | 届かない | 届かない |
| 診断（窓の雑音なし）、A、$p=10^{-4}$ / $3\times10^{-4}$ / $10^{-3}$ | 1.6 / 5.5 / 届かない | 5.6 / 17 / 届かない | 7.1 / 17 / 届かない | 6.0 / 4.2 / 届かない |
| 診断（窓の雑音なし）、B、$p=10^{-4}$ / $3\times10^{-4}$ / $10^{-3}$ | 0.52 / 1.6 / 16 | 2.7 / 16 / 33 | 17 / 12 / 13 | 19 / 5.0 / 未確定 |

「届かない」はこのモデルと探索範囲（$d\le61$）で届かないこと（方式として不可能という意味ではない）、「未確定」はブートストラップで判定できないこと。多くの値は $d>9$ や低い $p$ への外挿で、不確かさ（16〜84%）と外挿の区別は `results/cost_comparison.csv` の列に示す。

**3. 結論**：本研究の回路・復号器とモデルの範囲では、実用的な要求精度（$10^{-6}$ 以下）で魔法状態 1 個を作る時空間体積は Litinski 方式（15-to-1 蒸留）の方が小さく、切り替えの窓の雑音を除いた診断でもその傾向は残る。ただし診断は最適な復号器の限界ではなく、外挿にも依存するので、**最適化された Kobayashi 方式に対する一般的な優劣を示すものではない**。結論を変えうる要因（窓を扱える JIT 復号器、スケジュールの最適化、CCZ の実際の誤り率、FSC→パッチ変換、Litinski 側の注入と回路レベルの評価）は `docs/methods.md` §7–8。

## 版
v4（雑音 0 での回路構造と資源の比較）は git の履歴（コミット `049ce8f`）、v5.0（第 3 回レビュー前）は `0907464` に残してある。変更の履歴と外部レビューへの対応は `docs/methods.md` §9–10。
