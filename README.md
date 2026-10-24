# D4符号の code switching（Kobayashi方式）と 15-to-1 蒸留（Litinski方式）の時空間コスト比較 — 理想条件（物理ノイズ0）

論理魔法状態 $|T_L\rangle=T_L|+_L\rangle$ を 1 個作る 2 つのプロトコルを、**同じ計数ルール**でゲートレベルまで構成し、物理量子ビット数・時間・時空間体積を比較する。

* **Kobayashi方式**：folded surface code（FSC）→ D4（twisted $\mathbb Z_2^3$）Clifford 安定化子符号への code switching → transversal $U=WV$（論理 $T^\dagger$）→ FSC へ復帰（Kobayashi–Zhu–Hsin, arXiv:2511.02900）
* **Litinski方式**：*A Game of Surface Codes* Sec. 3 の 15-to-1 蒸留（11 tiles × 11τ の格子手術配置）

設計方針は [`docs/01_design_plan.md`](docs/01_design_plan.md)、仮定・未解決問題は [`docs/02_assumptions_and_open_issues.md`](docs/02_assumptions_and_open_issues.md)、ノイズ拡張は [`docs/03_noise_extension.md`](docs/03_noise_extension.md)。

## ファイル構成と実行順

| 順 | ノートブック | 役割 | 所要時間（目安） |
|---|---|---|---|
| – | `common_lib.ipynb`, `d4_lib.ipynb`, `msd_lib.ipynb` | 関数定義のみ（他のノートから `%run -i` で読み込む） | – |
| 1 | `D4_switching_main.ipynb` | D4 符号・code switching の構成と段階別資源計数 | 約 1 分 |
| 2 | `D4_switching_validation.ipynb` | D4 側の正当性検証（65 項目） | 約 6 分 |
| 3 | `MSD_15to1_main.ipynb` | 15-to-1 の tile モデル・物理モデルの資源計数 | 数十秒 |
| 4 | `MSD_15to1_validation.ipynb` | 15-to-1 側の正当性検証（31 項目） | 数十秒 |
| 5 | `resource_comparison.ipynb` | 比較表・グラフ・損益分岐・仮定一覧（1–4 の出力を読む） | 数十秒 |

出力：`results/`（CSV・JSON）、`figures/`（PNG）。各ノートの `QUICK = True` で小規模な動作確認ができる。乱数シードは `SEED = 20261024`。

### Google Colab での実行
1. すべての `.ipynb` を同じフォルダ（例：Google Drive の `MyDrive/D4_vs_MSD`）に置く。
2. 各ノートの最初のセルのコメントに従い、そのフォルダを作業ディレクトリにする（`drive.mount` → `os.chdir`）。
3. 上の順に上から実行する。`stim` などが無ければ 2 番目のセルが自動でインストールする（numpy, scipy, sympy, pandas, matplotlib は Colab 標準）。

## 主な結果（既定パラメータ、`FT_ref` モデル、比較A、データ＋測定量子ビット）

`FT_ref`：Litinski は文献どおり 1τ = $d$ cycles。Kobayashi は [KZH] が $O(d)$ としか述べていない段階を $\kappa d$ ラウンドとし **$\kappa=1$ を仮定**。物理時間は全ゲート 1 単位の仮定で、下界〜上界で示す。

| $d$ | ピーク量子ビット K / L | ラウンド K / L（別単位） | 物理時間 K / L（下界–上界） | 体積 qubit·round K / L | 体積比（qubit·time、下界同士 / 上界同士） |
|---|---|---|---|---|---|
| 5 | 319 / 545 | 15 / 55 | 133–288 / 342–672 | 3,785 / 29,975 | 0.18 / 0.21 |
| 9 | 1,007 / 1,777 | 27 / 99 | 233–512 / 606–1,200 | 21,357 / 175,923 | 0.17 / 0.20 |
| 15 | 2,759 / 4,945 | 45 / 165 | 383–848 / 1,002–1,992 | 97,155 / 815,925 | 0.17 / 0.20 |

厳密に同定した多項式（評価した全ての $d$ で一致）：
* ピーク量子ビット：Kobayashi $12d^2+4d-1$、Litinski $22d^2-5$（tile モデルでは $22d^2$）
* 体積 qubit·round：Kobayashi $(8\kappa_0+12\kappa_{in}+8\kappa_{out})d^3+\dots$（$\kappa=1$ で $28d^3+12d^2-3d$）、Litinski $242d^3-55d$（tile モデル・データのみで文献値 $121d^3$ を再現）

注意すべき点
* **損益分岐**：switch-in のラウンド数の定数 $\kappa_{in}$ が、qubit·round 体積で約 15〜19、物理時間体積で約 6〜13 を超えると Kobayashi 方式の体積が上回る（`results/comparison_breakeven_kappa.csv`）。$\kappa$ は文献から確定できていない。
* **比較B**：Kobayashi 方式は物理 CCZ を $6(d-1)^2$ 個／ラウンド使い、$d=9$ で CCZ 3,456 個（Clifford+T 分解で T 24,645 個）。Litinski 方式は物理 T 15 個。さらに FSC→通常パッチの変換が**未解決**のため、比較Bの Kobayashi の値は下界。
* **限界**：ノイズ0では 15-to-1 蒸留は本来不要であり、この比較は「最適な $T$ ゲート実装法」を決めるものではない（回路構造と資源の比較）。

## 正当性の検証（要約）
* D4：境界付き符号の安定化子が [KZH] Fig. 4 と一致、$C_Z$ 上で可換・$k=1$、$d_Z=L$、$U$ が論理 $T^\dagger$（$L\le6$ 局所厳密＋$L=1$ 独立実装）、プロトコル全体（ランダム測定結果と補正を含む）が $L\le2$ で忠実度 1、$L\le8$ で局所厳密証明、測定ガジェットとスケジュールの妥当性。
* 15-to-1：回路と行列の整合、理想入力→$T^\dagger|+\rangle$、誤り検出（$35p^3$）、auto-corrected π/8 の全分岐、11 tiles × 11τ と $121d^3$、stim による物理レベルの格子手術。
* 未検証：$L\ge3$ の D4 相の Knill–Laflamme 距離、耐故障性（JIT decoder、$\kappa$ の値）、FSC→通常パッチ変換。
