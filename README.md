# D4符号の code switching（Kobayashi方式）と 15-to-1 蒸留（Litinski方式）の時空間コスト比較 — 理想条件（物理ノイズ0）

論理魔法状態 $|T_L\rangle=T_L|+_L\rangle$ を 1 個作る 2 つのプロトコルを、**同じ計数ルール**でゲートレベルまで構成し、物理量子ビット数・時間・時空間体積を比較する。

* **Kobayashi方式**：folded surface code（FSC）→ D4（twisted $\mathbb Z_2^3$）Clifford 安定化子符号への code switching → transversal $U=WV$（論理 $T^\dagger$）→ FSC へ復帰（Kobayashi–Zhu–Hsin, arXiv:2511.02900）
* **Litinski方式**：*A Game of Surface Codes* Sec. 3 の 15-to-1 蒸留（11 tiles × 11τ の格子手術配置）

設計方針は [`docs/01_design_plan.md`](docs/01_design_plan.md)、仮定・未解決問題は [`docs/02_assumptions_and_open_issues.md`](docs/02_assumptions_and_open_issues.md)、ノイズ拡張は [`docs/03_noise_extension.md`](docs/03_noise_extension.md)、外部レビューへの対応は [`docs/04_review_response.md`](docs/04_review_response.md)。

## ファイル構成と実行順

| 順 | ノートブック | 役割 | 所要時間（目安） |
|---|---|---|---|
| – | `common_lib.ipynb`, `d4_lib.ipynb`, `msd_lib.ipynb` | 関数定義のみ（他のノートから `%run -i` で読み込む） | – |
| 1 | `D4_switching_main.ipynb` | D4 符号・code switching の構成と段階別資源計数 | 約 1 分 |
| 2 | `D4_switching_validation.ipynb` | D4 側の正当性検証（V1–V11、V5b・V7b を含む） | 約 10 分 |
| 3 | `MSD_15to1_main.ipynb` | 15-to-1 の tile モデル・物理モデルの資源計数 | 数十秒 |
| 4 | `MSD_15to1_validation.ipynb` | 15-to-1 側の正当性検証（M1–M9、多パッチ測定の物理レベル構成を含む） | 約 2 分 |
| 5 | `resource_comparison.ipynb` | 比較表・グラフ・損益分岐・仮定一覧（1–4 の出力を読む） | 数十秒 |

出力：`results/`（CSV・JSON）、`figures/`（PNG）。各ノートの `QUICK = True` で小規模な動作確認ができる。乱数シードは `SEED = 20261024`。

### Google Colab での実行
1. すべての `.ipynb` を同じフォルダ（例：Google Drive の `MyDrive/D4_vs_MSD`）に置く。
2. 各ノートの最初のセルのコメントに従い、そのフォルダを作業ディレクトリにする（`drive.mount` → `os.chdir`）。
3. 上の順に上から実行する。`stim` などが無ければ 2 番目のセルが自動でインストールする（numpy, scipy, sympy, pandas, matplotlib は Colab 標準）。

## 主な結果（**予備的評価**：特定の仮定のもとでの値）

> この節の数値は、Kobayashi 側の $O(d)$ ラウンドの定数 $\kappa$（**文献で未確定、既定値 1 は仮定**）、全ゲート 1 単位の所要時間（仮定）、未構成の FSC→通常パッチ変換に依存する。Kobayashi 方式の実装上の優位性を示す確定的な結果ではない。研究上の主な結果は、単一の比よりも下の**損益分岐と感度**である。

`FT_ref`、比較A、データ＋測定量子ビット。Litinski は 1τ = $d$ cycles（文献）、Kobayashi は $\kappa_0=\kappa_{in}=\kappa_{out}=1$（仮定）。横軸 $L$ は FSC（入出力の符号）の距離 $d$ に等しい。D4 相の距離も $L$ であることは $L\le5$（6）で証明済み。

| $L=d$ | ピーク量子ビット K / L | ラウンド K / L（別単位） | 物理時間 K / L（下界–上界） | 体積 qubit·round K / L | 体積比（qubit·time、下界同士 / 上界同士） |
|---|---|---|---|---|---|
| 5 | 319 / 544 | 15 / 55 | 133–288 / 342–672 | 3,785 / 29,920 | 0.18 / 0.21 |
| 9 | 1,007 / 1,776 | 27 / 99 | 233–512 / 606–1,200 | 21,357 / 175,824 | 0.17 / 0.20 |
| 15 | 2,759 / 4,944 | 45 / 165 | 383–848 / 1,002–1,992 | 97,155 / 815,760 | 0.17 / 0.20 |

厳密に同定した多項式（評価した全ての $d$ で一致）：
* ピーク量子ビット：Kobayashi $12d^2+4d-1$、Litinski $22d^2-6$（11 回の多パッチ測定を明示的に構成して計数；tile モデルでは $22d^2$）
* 体積 qubit·round：Kobayashi $(8\kappa_0+12\kappa_{in}+8\kappa_{out})d^3+\dots$（$\kappa=1$ で $28d^3+12d^2-3d$）、Litinski $242d^3-66d$（tile モデル・データのみで文献値 $121d^3$ を再現）

**損益分岐と感度（主結果）**
* switch-in のラウンド数 $\lceil\kappa_{in}d\rceil$ の定数が、qubit·round 体積で約 15〜19、物理時間体積で約 6〜13 を超えると Kobayashi 方式の体積が上回る（`results/comparison_breakeven_kappa.csv`）。
* CCZ の所要時間が長いほど損益分岐は下がる。$L=15$ で物理時間体積の損益分岐 $\kappa_{in}$ は、$t_{\rm CCZ}=1$ で約 8〜12、$t_{\rm CCZ}=5$ で約 4〜7、$t_{\rm CCZ}=12$ で約 2〜4（上界同士〜下界同士；`results/comparison_sensitivity_tCCZ.csv`、`figures/comparison_sensitivity_kappa_tCCZ.png`）。

**比較B（未完成であることに注意）**
* Kobayashi 方式は物理 CCZ を $6(d-1)^2$ 個／ラウンド使う（$d=9$ で 3,456 個、Clifford+T 分解で T 24,645 個）。Litinski 方式は物理 T 15 個。
* FSC→通常パッチ変換は**未解決**で、合計には含めず「未計上コスト」（`results/uncounted_costs.csv`）に分離した。比較Bの Kobayashi の合計は下界であり、共通のスタートとゴールからの完全な比較にはなっていない。

**限界**：ノイズ0では 15-to-1 蒸留は本来不要であり、この比較は「最適な $T$ ゲート実装法」を決めるものではない（回路構造と資源の比較）。

## 正当性の検証（要約）と、検証が保証しないこと
**検証したこと**
* D4：境界付き符号の安定化子が [KZH] Fig. 4 と一致。$C_Z$ 上で可換・$k=1$。$d_Z=L$。D4 相の Knill–Laflamme 距離 $=L$（$L\le5$、`FULL_KL=True` で 6）。$U$ が論理 $T^\dagger$（$L\le6$ の局所厳密検証＋$L=1$ の独立実装）。プロトコル全体については意味の異なる 3 種類の検証を行った：符号状態の厳密シミュレーション（$L\le2$）、実回路と位相公式の照合（$L\le4$）、位相公式の局所厳密証明（$L\le8$）。測定ガジェットとスケジュールの妥当性（演算子レベル）。
* 15-to-1：回路と行列の整合、理想入力→$T^\dagger|+\rangle$、誤り検出（$35p^3$）、auto-corrected π/8 の全分岐、11 tiles × 11τ と $121d^3$。Fig. 17c の 11 回の多パッチ測定 $P\otimes Z_m$ を物理（検査）レベルで構成し、$\mathbb F_2$ 厳密判定・合併符号の距離・stim で検証。

**検証が保証しないこと**（PASS の数はこれらを保証しない）
* 耐故障性（JIT decoder の性能、$\kappa$ の値）：ノイズを入れていない。
* FSC→通常パッチ変換（未構成）、$Z_m\otimes Y_0$ の twist 継ぎ目（未構成）。
* $L\ge6$（または 7）での D4 相の距離、$L\ge3$ での符号状態全体の直接シミュレーション。
* ハードウェアに即したゲート時間・接続性（D4 の測定補助は最大 9 量子ビットと相互作用する）。
